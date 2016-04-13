#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"
#include "wavelet.hpp"

struct node_t;
struct superroot_t;

using node_ptr_t            = boost::interprocess::offset_ptr<node_t>;
using superroot_ptr_t       = boost::interprocess::offset_ptr<superroot_t>;

using segment_manager_t     = boost::interprocess::managed_mapped_file::segment_manager;
using allocator_superroot_t = boost::interprocess::allocator<superroot_ptr_t, segment_manager_t>;
using superroot_vector_t    = boost::interprocess::vector<superroot_ptr_t, allocator_superroot_t>;

struct superroot_t {
    node_ptr_t root;
    calc_t     approx;
    calc_t     error;
};

struct node_t {
    node_ptr_t child_l;
    node_ptr_t child_r;
    calc_t     x;
};

using mapped_file_ptr_t = std::shared_ptr<boost::interprocess::managed_mapped_file>;

template <typename T>
boost::interprocess::offset_ptr<T> alloc_in_mapped_file(mapped_file_ptr_t& f) {
    return static_cast<T*>(f->allocate(sizeof(T)));
}

template <typename T>
void dealloc_in_mapped_file(mapped_file_ptr_t& f, const boost::interprocess::offset_ptr<T>& ptr) {
    f->deallocate(ptr.get());
}

namespace std {
    template<>
    struct hash<node_ptr_t> {
        size_t operator()(const node_ptr_t& obj) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, obj.get());
            return seed;
        }
    };

    template<>
    struct hash<std::pair<node_ptr_t, node_ptr_t>> {
        size_t operator()(const std::pair<node_ptr_t, node_ptr_t>& obj) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, obj.first.get());
            boost::hash_combine(seed, obj.second.get());
            return seed;
        }
    };
}

class range_bucket_t {
    public:
        range_bucket_t() = default;

        void insert(node_ptr_t node) {
            auto lower = std::lower_bound(_slot.begin(), _slot.end(), node, [](node_ptr_t a, node_ptr_t b){
                return a->x < b->x;
            });
            _slot.insert(lower, node);
        }

        std::vector<std::pair<node_ptr_t, double>> get_nearest(node_ptr_t node, double max_dist, std::size_t max_size) {
            auto begin  = _slot.begin();
            auto end    = _slot.end();
            auto center = std::lower_bound(begin, end, node, [](node_ptr_t a, node_ptr_t b) {
                return a->x < b->x;
            });

            std::vector<std::pair<node_ptr_t, double>> neighbors;
            auto it_up       = center;
            auto it_down     = std::prev(center);
            double dist_up   = std::numeric_limits<double>::infinity();
            double dist_down = std::numeric_limits<double>::infinity();
            auto prev_begin  = std::prev(begin); // let's also hope that std::prev(begin) works
            if (it_up != end) {
                dist_up = std::abs((*it_up)->x - node->x);
            }
            if (it_down != prev_begin) {
                dist_down = std::abs((*it_down)->x - node->x);
            }
            while ((neighbors.size() < max_size)
                    && (((it_up != end) && (dist_up <= max_dist)) || ((it_down != prev_begin) && (dist_down <= max_dist)))) {
                if (dist_up < dist_down) {
                    neighbors.push_back(std::make_pair(*it_up, dist_up));
                    ++it_up;
                    if (it_up != end) {
                        dist_up = std::abs((*it_up)->x - node->x);
                    } else {
                        dist_up = std::numeric_limits<double>::infinity();
                    }
                } else {
                    neighbors.push_back(std::make_pair(*it_down, dist_down));
                    --it_down;
                    if (it_down != prev_begin) {
                        dist_down = std::abs((*it_down)->x - node->x);
                    } else {
                        dist_down = std::numeric_limits<double>::infinity();
                    }
                }
            }

            return neighbors;
        }

        void delete_all_ptrs(mapped_file_ptr_t& f) {
            for (auto& node : _slot) {
                dealloc_in_mapped_file(f, node);
            }
        }

    private:
        std::vector<node_ptr_t> _slot;
};

class range_index_t {
    public:
        range_index_t() = default;

        range_bucket_t& get_bucket(const node_ptr_t node) {
            return _index_struct[std::make_pair(node->child_l, node->child_r)];
        }

        void delete_all_ptrs(mapped_file_ptr_t& f) {
            for (auto& kv : _index_struct) {
                kv.second.delete_all_ptrs(f);
            }
        }

    private:
        // XXX: make node pointers in key const
        std::unordered_map<std::pair<node_ptr_t, node_ptr_t>, range_bucket_t> _index_struct;
};

struct index_t {
    superroot_vector_t* superroots;
    std::vector<std::vector<range_index_t>> levels;
    std::size_t node_count;

    index_t(superroot_vector_t* superroots_, std::size_t depth) : superroots(superroots_), levels(depth), node_count(0) {
        for (std::size_t l = 0; l < depth; ++l) {
            levels[l] = std::vector<range_index_t>(1 << l);
        }
    }

    void delete_all_ptrs(mapped_file_ptr_t& f) {
        for (auto& sr : *superroots) {
            if (sr != nullptr) {
                dealloc_in_mapped_file(f, sr);
            }
        }
        for (auto& level : levels) {
            for (auto& range_index : level) {
                range_index.delete_all_ptrs(f);
            }
        }
    }
};

namespace po = boost::program_options;

constexpr std::size_t power_of_2(std::size_t x) {
    std::size_t p = 0;
    while (x > 1) {
        x = x >> 1;
        ++p;
    }
    return p;
}

class transformer {
    public:
        superroot_ptr_t                      superroot;
        std::vector<std::vector<node_ptr_t>> levels;

        transformer(std::size_t ylength, std::size_t depth, const mapped_file_ptr_t& mapped_file) :
            superroot(nullptr),
            levels(depth),
            _ylength(ylength),
            _depth(depth),
            _mywt(std::make_shared<wavelet>("haar"), "dwt", static_cast<int>(_ylength), static_cast<int>(_depth)),
            _influence_sqrt(_depth),
            _mapped_file(mapped_file)
        {
            _mywt.extension("per");
            _mywt.conv("direct");

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t influence = 1u << (_depth - l);
                _influence_sqrt[l] = std::sqrt(static_cast<double>(influence));
            }
        }

        superroot_ptr_t data_to_tree(const calc_t* data) {
            _mywt.run_dwt(data);

            superroot = alloc_in_mapped_file<superroot_t>(_mapped_file);
            superroot->approx = _mywt.output()[0];
            superroot->error  = 0;

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t outdelta  = 1 << l;
                std::size_t width     = outdelta;  // same calculation

                levels[l].clear();

                for (std::size_t idx = 0; idx < width; ++idx) {
                    node_ptr_t node  = alloc_in_mapped_file<node_t>(_mapped_file);
                    node->child_l = nullptr;
                    node->child_r = nullptr;
                    node->x       = _mywt.output()[outdelta + idx] * _influence_sqrt[l];

                    link_to_parent(node, l, idx);

                    levels[l].push_back(node);
                }
            }

            return superroot;
        }

        void tree_to_data(calc_t* data) {
            // prepare idwt
            _mywt.output()[0] = superroot->approx;
            std::vector<node_ptr_t> layer_a{superroot->root};
            std::vector<node_ptr_t> layer_b;
            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t width    = static_cast<std::size_t>(1) << l;
                std::size_t outdelta = width; // same calculation

                layer_b.clear();
                for (std::size_t idx = 0; idx < width; ++idx) {
                    _mywt.output()[outdelta + idx] = layer_a[idx]->x / _influence_sqrt[l];
                    layer_b.push_back(layer_a[idx]->child_l);
                    layer_b.push_back(layer_a[idx]->child_r);
                }

                std::swap(layer_a, layer_b);
            }

            // do idwt
            _mywt.run_idwt(data);
        }

        /*
         * link specified node to the right parent above it (calculated by using l and idx)
         */
        void link_to_parent(node_ptr_t node, std::size_t l, std::size_t idx) {
            if (l == 0) {
                superroot->root = node;
            } else {
                auto parent = levels[l - 1][idx >> 1];

                if (idx & 1u) {
                    parent->child_r = node;
                } else {
                    parent->child_l = node;
                }
            }
        }

    private:
        const std::size_t   _ylength;
        const std::size_t   _depth;
        wavelet_transform   _mywt;
        std::vector<double> _influence_sqrt;
        mapped_file_ptr_t   _mapped_file;
};

class engine {
    public:
        engine(std::size_t ylength, std::size_t depth, double max_error, const calc_t* base, index_t* index, const mapped_file_ptr_t& mapped_file)
            : _ylength(ylength),
            _depth(depth),
            _max_error(max_error),
            _base(base),
            _index(index),
            _mapped_file(mapped_file),
            _transformer(_ylength, _depth, _mapped_file) {}

        superroot_ptr_t run(std::size_t i) {
            _transformer.data_to_tree(_base + (i * _ylength));
            _transformer.superroot->error = calc_error(i);  // correct error because of floating point errors
            (*_index->superroots)[i] = _transformer.superroot;
            run_mergeloop(i);
            drain();
            _transformer.superroot->error = calc_error(i);  // correct error one last time

            return _transformer.superroot;
        }

    private:
        struct queue_entry_t {
            double      dist;
            std::size_t l;
            std::size_t idx;
            node_ptr_t     neighbor;

            queue_entry_t(double dist_, std::size_t l_, std::size_t idx_, node_ptr_t neighbor_) :
                dist(dist_),
                l(l_),
                idx(idx_),
                neighbor(neighbor_) {}
        };

        struct queue_entry_compare {
            bool operator()(const queue_entry_t& a, const queue_entry_t& b) {
                return a.dist > b.dist;
            }
        };

        using queue_t = std::priority_queue<queue_entry_t, std::vector<queue_entry_t>, queue_entry_compare>;

        const std::size_t                _ylength;
        const std::size_t                _depth;
        double                           _max_error;
        const calc_t*                    _base;
        index_t*                         _index;
        std::mt19937                     _rng;
        mapped_file_ptr_t                _mapped_file;
        transformer                      _transformer;

        void run_mergeloop(std::size_t i) {
            queue_t queue;
            bool error_is_approx = false;

            // fill queue with entries from lowest level
            std::vector<std::size_t> indices(_transformer.levels[_depth - 1].size());
            for (std::size_t idx = 0; idx < indices.size(); ++idx) {
                indices[idx] = idx;
            }
            std::shuffle(indices.begin(), indices.end(), _rng);
            for (auto idx : indices) {
                std::size_t l = _depth - 1;
                generate_queue_entries(l, idx, queue);
            }

            // now run merge loop
            while (!queue.empty()) {
                auto best = queue.top();
                queue.pop();

                node_ptr_t current_node = _transformer.levels[best.l][best.idx];

                // check if node was already merged and if distance is low enough
                if (current_node != nullptr) {
                    // check if merge would be in error range (i.e. does not increase error beyond _max_error)
                    bool in_error_range = false;
                    if (_transformer.superroot->error + best.dist < _max_error) {
                        in_error_range = true;
                    } else if (error_is_approx) {
                        // ok, error is an approximation anyway, so lets calculate the right value and try again
                        _transformer.superroot->error = calc_error(i);
                        error_is_approx = false;
                        if (_transformer.superroot->error + best.dist < _max_error) {
                            in_error_range = true;
                        }
                    }

                    if (in_error_range) {
                        // calculate approx. error
                        _transformer.superroot->error += best.dist;
                        error_is_approx = true;

                        // execute merge
                        _transformer.link_to_parent(best.neighbor, best.l, best.idx);

                        // remove old node and mark as merged
                        dealloc_in_mapped_file(_mapped_file, current_node);
                        _transformer.levels[best.l][best.idx] = nullptr;

                        // generate new queue entries
                        std::size_t idx_even = best.idx - (best.idx % 2);
                        if ((best.l > 0) && (_transformer.levels[best.l][idx_even] == nullptr) && (_transformer.levels[best.l][idx_even + 1] == nullptr)) {
                            generate_queue_entries(best.l - 1, best.idx >> 1, queue);
                        }
                    }
                }
            }
        }

        void generate_queue_entries(std::size_t l, std::size_t idx, queue_t& queue) {
            node_ptr_t current_node = _transformer.levels[l][idx];

            if (current_node != nullptr) {
                auto& current_index_slot = _index->levels[l][idx];
                auto& bucket = current_index_slot.get_bucket(current_node);
                auto neighbors = bucket.get_nearest(current_node, _max_error - _transformer.superroot->error, 1);
                if (!neighbors.empty()) {
                    queue.emplace(neighbors[0].second, l, idx, neighbors[0].first);
                }
            }
        }

        /*
         * adds all remaining to index, without any merges
         */
        void drain() {
            // do work bottom to top, might be important for some transaction implementaions later
            for (std::size_t l_plus = _depth; l_plus > 0; --l_plus) {
                std::size_t l = l_plus - 1;

                for (std::size_t idx = 0; idx < _transformer.levels[l].size(); ++idx) {
                    node_ptr_t current_node = _transformer.levels[l][idx];

                    // node might already be merged
                    if (current_node != nullptr) {
                        auto& current_index_slot = _index->levels[l][idx];
                        auto& bucket = current_index_slot.get_bucket(current_node);
                        bucket.insert(current_node);
                        ++_index->node_count;
                    }
                }
            }
        }

        double calc_error(const std::size_t i) {
            // do idwt
            std::vector<double> data_approximated(_ylength);
            _transformer.tree_to_data(data_approximated.data());

            // compare
            double error = 0.0;
            const calc_t* local_base = _base + (i * _ylength);
            for (std::size_t y = 0; y < _ylength; ++y) {
                error += std::abs(data_approximated[y] - local_base[y]);
            }

            return error;
        }
};

class printer {
    public:
        printer(superroot_vector_t* superroots, const std::shared_ptr<idx_ngram_map_t>& idxmap) :
            _superroots(superroots),
            _idxmap(idxmap) {}

        std::ostream& print_begin(std::ostream &out) {
            out << "digraph treegraph {" << std::endl;
            out << "  nodesep=0.1;" << std::endl;
            out << "  ranksep=5;" << std::endl;
            out << "  size=\"25,25\";" << std::endl;
            out << std::endl;

            return out;
        }

        std::ostream& print_end(std::ostream &out) {
            out << "}" << std::endl;

            return out;
        }

        std::ostream& print_tree(std::ostream &out, std::size_t i) {
            superroot_ptr_t superroot = (*_superroots)[i];
            print_superroot(out, superroot, (*_idxmap)[i]);

            std::vector<node_ptr_t> todo;
            todo.push_back(superroot->root);

            while (!todo.empty()) {
                node_ptr_t current = todo.back();
                todo.pop_back();

                print_node(out, current);

                if (current->child_l) {
                    todo.push_back(current->child_l);
                }
                if (current->child_r) {
                    todo.push_back(current->child_r);
                }
            }

            return out;
        }

    private:
        superroot_vector_t*              _superroots;
        std::shared_ptr<idx_ngram_map_t> _idxmap;
        std::unordered_set<node_ptr_t>   _printed_nodes;

        template <typename T>
        std::ostream& print_address(std::ostream &out, T* addr) {
            out << "addr" << addr;
            return out;
        }

        template <typename T>
        std::ostream& print_address(std::ostream &out, boost::interprocess::offset_ptr<T> addr) {
            return print_address(out, addr.get());
        }

        std::ostream& print_superroot(std::ostream &out, superroot_ptr_t superroot, const ngram_t& ngram) {
            out << "  ";
            print_address(out, superroot);
            out << " [label=\"" << boost::locale::conv::utf_to_utf<char>(ngram) << "\", shape=circle, fixedsize=true, width=6, height=6, fontsize=80];" << std::endl;

            out << "  ";
            print_address(out, superroot);
            out << " -> ";
            print_address(out, superroot->root);
            out << ";" << std::endl;

            out << std::endl;

            return out;
        }

        std::ostream& print_node(std::ostream &out, node_ptr_t node) {
            if (_printed_nodes.find(node) == _printed_nodes.end()) {
                out << "  ";
                print_address(out, node);
                out << " [label=\"" << node->x << "\", shape=box, fixedsize=true, width=1.0, height=0.5, fontsize=8];" << std::endl;

                if (node->child_l) {
                    out << "  ";
                    print_address(out, node);
                    out << " -> ";
                    print_address(out, node->child_l);
                    out << ";" << std::endl;
                }
                if (node->child_r) {
                    out << "  ";
                    print_address(out, node);
                    out << " -> ";
                    print_address(out, node->child_r);
                    out << ";" << std::endl;
                }

                out << std::endl;

                _printed_nodes.insert(node);
            }

            return out;
        }
};

double calc_compression_rate(const index_t* index, year_t ylength, std::size_t n) {
    std::size_t size_normal = sizeof(calc_t) * static_cast<std::size_t>(ylength) * n;
    std::size_t size_compression = sizeof(node_t) * index->node_count + sizeof(superroot_t) * n;

    return static_cast<double>(size_compression) / static_cast<double>(size_normal);
}

void print_process(const index_t* index, year_t ylength, std::size_t n, std::size_t i) {
    std::cout << "  process=" << i << "/" << n
        << " #nodes=" << index->node_count
        << " %nodes=" << (static_cast<double>(index->node_count) / static_cast<double>((ylength - 1) * i))
        << " compression_rate=" << calc_compression_rate(index, ylength, i)
        << std::endl;
}

int main(int argc, char** argv) {
    std::string fname_binary;
    std::string fname_map;
    std::string fname_dot;
    std::string fname_index;
    std::size_t index_size;
    year_t ylength;
    double max_error;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("error", po::value(&max_error)->required(), "maximum error during tree merge")
        ("dotfile", po::value(&fname_dot), "dot file that represents the index (optional)")
        ("index", po::value(&fname_index)->required(), "index file")
        ("size", po::value(&index_size)->required(), "size of the index file (in bytes)")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "index_dtw")) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }

    auto idxmap = std::make_shared<idx_ngram_map_t>();
    ngram_idx_map_t ngmap;
    std::tie(*idxmap, ngmap) = parse_map_file(fname_map);
    std::size_t n = ngmap.size();

    boost::iostreams::mapped_file_params params;
    params.path   = fname_binary;
    params.flags  = boost::iostreams::mapped_file::mapmode::readonly;
    params.length = static_cast<std::size_t>(n * ylength * sizeof(calc_t));
    params.offset = 0;
    boost::iostreams::mapped_file input(params);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    std::cout << "open index file..." << std::endl;
    auto findex = std::make_shared<boost::interprocess::managed_mapped_file>(
        boost::interprocess::create_only,
        fname_index.c_str(),
        index_size
    );
    auto segment_manager = findex->get_segment_manager();
    allocator_superroot_t allocator_superroot(segment_manager);
    auto superroots = findex->construct<superroot_vector_t>("superroots")(n, allocator_superroot);
    std::cout << "done" << std::endl;

    std::cout << "build and merge trees" << std::endl;
    std::size_t depth = power_of_2(ylength);
    index_t index(superroots, depth);
    engine eng(ylength, depth, max_error, base, &index, findex);
    std::mt19937 rng;
    std::size_t counter = 0;
    std::vector<std::size_t> indices(n);
    std::generate(indices.begin(), indices.end(), [&counter](){
        return counter++;
    });
    std::shuffle(indices.begin(), indices.end(), rng);

    std::size_t n_test = 10000; // DEBUG
    for (std::size_t i = 0; i < n_test; ++i) {
        if (i % 1000 == 0) {
            print_process(&index, ylength, n_test, i);
        }
        eng.run(indices[i]);
    }
    print_process(&index, ylength, n_test, n_test);
    std::cout << "done" << std::endl;

    if (vm.count("dotfile")) {
        std::ofstream out(fname_dot);
        printer pr(superroots, idxmap);
        pr.print_begin(std::cout);
        for (std::size_t i = 0; i < 100; ++i) {
            pr.print_tree(std::cout, i);
        }
        pr.print_end(std::cout);
    }

    // free memory for sanity checking
    index.delete_all_ptrs(findex);
}
