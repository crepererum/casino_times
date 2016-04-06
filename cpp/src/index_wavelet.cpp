#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"
#include "wavelet.hpp"

struct node_t;
struct superroot_t;

struct superroot_t {
    ngram_t ngram;
    node_t* root;
    calc_t  approx;
    calc_t  error;
};

struct node_t {
    node_t* child_l;
    node_t* child_r;
    calc_t  x;
};

namespace std {
    template<>
    struct hash<std::pair<node_t*, node_t*>> {
        size_t operator()(const std::pair<node_t*, node_t*>& obj) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, obj.first);
            boost::hash_combine(seed, obj.second);
            return seed;
        }
    };
}

class range_bucket_t {
    public:
        range_bucket_t() = default;

        void insert(node_t* node) {
            auto lower = std::lower_bound(_slot.begin(), _slot.end(), node, [](node_t* a, node_t* b){
                return a->x < b->x;
            });
            _slot.insert(lower, node);
        }

        std::pair<node_t*, double> get_nearest(node_t* node) {
            auto begin       = _slot.begin();
            auto end         = _slot.end();
            auto lower       = std::lower_bound(begin, end, node, [](node_t* a, node_t* b) {
                return a->x < b->x;
            });

            // work around edge cases (e.g. lower == begin or lower == end)
            if (lower != begin) {
                // there is an element before the lower bound
                auto lower_minus        = std::prev(lower);
                double dist_lower_minus = std::abs((*lower_minus)->x - node->x);

                // is this the end?
                if (lower != end) {
                    // also the lower element exists (for real)
                    double dist_lower = std::abs((*lower)->x - node->x);

                    // now figure out which is the nearest element
                    if (dist_lower < dist_lower_minus) {
                        return std::make_pair(*lower, dist_lower);
                    } else {
                        return std::make_pair(*lower_minus, dist_lower_minus);
                    }
                } else {
                    // this is the end!
                    // so the element before lower must be the nearest neighbor
                    return std::make_pair(*lower_minus, dist_lower_minus);
                }
            } else {
                // is this the end?
                if (lower != end) {
                    // nope, cool, but because lower is the first element, it's also the only
                    // candidate for the nearest neighbor
                    double dist_lower = std::abs((*lower)->x - node->x);
                    return std::make_pair(*lower, dist_lower);
                } else {
                    // begin == end == lower
                    // so the slot was empty
                    return std::make_pair(nullptr, std::numeric_limits<double>::infinity());
                }
            }
        }

        void delete_all_ptrs() {
            for (auto& node : _slot) {
                delete node;
            }
        }

    private:
        std::vector<node_t*> _slot;
};

class range_index_t {
    public:
        range_index_t() = default;

        range_bucket_t& get_bucket(const node_t* node) {
            return _index_struct[std::make_pair(node->child_l, node->child_r)];
        }

        void delete_all_ptrs() {
            for (auto& kv : _index_struct) {
                kv.second.delete_all_ptrs();
            }
        }

    private:
        std::unordered_map<std::pair<node_t*, node_t*>, range_bucket_t> _index_struct;
};

struct index_t {
    std::vector<superroot_t*> superroots;
    std::vector<std::vector<range_index_t>> levels;
    std::size_t node_count;

    index_t(std::size_t depth) : levels(depth), node_count(0) {
        for (std::size_t l = 0; l < depth; ++l) {
            levels[l] = std::vector<range_index_t>(1 << l);
        }
    }

    void delete_all_ptrs() {
        for (auto& sr : superroots) {
            delete sr;
        }
        for (auto& level : levels) {
            for (auto& range_index : level) {
                range_index.delete_all_ptrs();
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
        transformer(std::size_t ylength, std::size_t depth, double max_error, const calc_t* base, const std::shared_ptr<idx_ngram_map_t>& idxmap, index_t* index)
            : _ylength(ylength),
            _depth(depth),
            _max_error(max_error),
            _mywt(std::make_shared<wavelet>("haar"), "dwt", static_cast<int>(_ylength), static_cast<int>(_depth)),
            _base(base),
            _idxmap(idxmap),
            _index(index),
            _levels(_depth)
        {
            _mywt.extension("per");
            _mywt.conv("direct");
        }

        superroot_t* run(std::size_t i) {
            run_dwt(i);
            superroot_t* superroot = transform_to_tree(i);
            add_to_index(superroot);

            return superroot;
        }

    private:
        const std::size_t                 _ylength;
        const std::size_t                 _depth;
        double                            _max_error;
        wavelet_transform                 _mywt;
        const calc_t*                     _base;
        std::shared_ptr<idx_ngram_map_t>  _idxmap;
        index_t*                          _index;
        std::vector<std::vector<node_t*>> _levels;
        std::mt19937                      _rng;

        void run_dwt(std::size_t i) {
            const calc_t* data = _base + (i * _ylength);
            _mywt.run_dwt(data);
        }

        superroot_t* transform_to_tree(std::size_t i) {
            superroot_t* superroot = new superroot_t;
            superroot->ngram  = (*_idxmap)[i];
            superroot->approx = _mywt.output()[0];
            superroot->error  = 0;

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t outdelta  = 1 << l;
                std::size_t width     = outdelta;  // same calculation
                std::size_t influence = 1u << (_depth - l);
                double influence_sqrt = std::sqrt(static_cast<double>(influence));  // XXX: precalc

                for (std::size_t idx = 0; idx < width; ++idx) {
                    node_t* node = new node_t;
                    node->child_l   = nullptr;
                    node->child_r   = nullptr;
                    node->x         = _mywt.output()[outdelta + idx] * influence_sqrt;

                    link_to_parent(node, l, idx, superroot);

                    _levels[l].push_back(node);
                }
            }

            return superroot;
        }

        void add_to_index(superroot_t* superroot) {
            _index->superroots.push_back(superroot);

            for (std::size_t l_plus = _depth; l_plus > 0; --l_plus) {
                std::size_t l = l_plus - 1;

                std::vector<range_bucket_t*> buckets(_levels[l].size());
                for (std::size_t idx = 0; idx < _levels[l].size(); ++idx) {
                    node_t* current_node = _levels[l][idx];
                    auto& current_index_slot = _index->levels[l][idx];
                    buckets[idx] = &current_index_slot.get_bucket(current_node);
                }
                // XXX: ensure that bucket pointers are stable during that function call

                // bucket lookup depends on the children, so we don't need to do check if children are equal here
                std::vector<std::pair<node_t*, double>> neighbors(_levels[l].size());
                for (std::size_t idx = 0; idx < _levels[l].size(); ++idx) {
                    node_t* current_node = _levels[l][idx];
                    neighbors[idx] = buckets[idx]->get_nearest(current_node);
                }

                // merge pairs with merged children first to give better changes of large sub-tree merges
                // INFO: "merged children" can also mean: no children at all (e.g. for the lowest level)
                if (l > 0) {
                    std::vector<std::pair<std::size_t, double>> indices_pairs;
                    for (std::size_t idx = 0; idx < _levels[l].size(); idx += 2) {
                        if (neighbors[idx].first != nullptr && neighbors[idx + 1].first != nullptr) {
                            indices_pairs.push_back(std::make_pair(idx, neighbors[idx].second + neighbors[idx + 1].second));
                        }
                    }
                    std::shuffle(indices_pairs.begin(), indices_pairs.end(), _rng);
                    std::sort(indices_pairs.begin(), indices_pairs.end(), [](const auto& a, const auto& b){
                        return a.second < b.second;
                    });
                    for (const auto& kv : indices_pairs) {
                        std::size_t idx = kv.first;
                        node_t* current_node_l = _levels[l][idx];
                        node_t* current_node_r = _levels[l][idx + 1];

                        if (superroot->error + kv.second < _max_error) {
                            superroot->error += kv.second;

                            link_to_parent(neighbors[idx    ].first, l, idx    , superroot);
                            link_to_parent(neighbors[idx + 1].first, l, idx + 1, superroot);

                            delete current_node_l;
                            delete current_node_r;

                            // mark them as merged
                            _levels[l][idx]     = nullptr;
                            _levels[l][idx + 1] = nullptr;
                        } else {
                            // they're sorted, so we can abort here
                            break;
                        }
                    }
                }

                // now merge remaining nodes one-by-one
                // keep ALL remaining nodes, otherwise they won't be added to the buckets
                std::vector<std::pair<std::size_t, double>> indices_single;
                for (std::size_t idx = 0; idx < _levels[l].size(); ++idx) {
                    node_t* current_node = _levels[l][idx];

                    // node might already be merged
                    if (current_node != nullptr) {
                        indices_single.push_back(std::make_pair(idx, neighbors[idx].second));
                    }
                }
                std::shuffle(indices_single.begin(), indices_single.end(), _rng);
                std::sort(indices_single.begin(), indices_single.end(), [](const auto& a, const auto& b){
                    return a.second < b.second;
                });
                for (const auto& kv : indices_single) {
                    std::size_t idx = kv.first;
                    node_t* current_node = _levels[l][idx];

                    if (neighbors[idx].first != nullptr && superroot->error + kv.second < _max_error) {
                        superroot->error += kv.second;
                        link_to_parent(neighbors[idx].first, l, idx, superroot);
                        delete current_node;
                    } else {
                        buckets[idx]->insert(current_node);
                        ++_index->node_count;
                    }
                }

                _levels[l].clear();
            }

        }

        void link_to_parent(node_t* node, std::size_t l, std::size_t idx, superroot_t* superroot) {
            if (l == 0) {
                superroot->root = node;
            } else {
                auto parent = _levels[l - 1][idx >> 1];

                if (idx & 1u) {
                    parent->child_r = node;
                } else {
                    parent->child_l = node;
                }
            }
        }
};

class printer {
    public:
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

        std::ostream& print_tree(std::ostream &out, superroot_t* superroot) {
            print_superroot(out, superroot);

            std::vector<node_t*> todo;
            todo.push_back(superroot->root);

            while (!todo.empty()) {
                node_t* current = todo.back();
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
        std::unordered_set<node_t*> _printed_nodes;

        template <typename T>
        std::ostream& print_address(std::ostream &out, T* addr) {
            out << "addr" << addr;
            return out;
        }

        std::ostream& print_superroot(std::ostream &out, superroot_t* superroot) {
            out << "  ";
            print_address(out, superroot);
            out << " [label=\"" << boost::locale::conv::utf_to_utf<char>(superroot->ngram) << "\", shape=circle, fixedsize=true, width=6, height=6, fontsize=80];" << std::endl;

            out << "  ";
            print_address(out, superroot);
            out << " -> ";
            print_address(out, superroot->root);
            out << ";" << std::endl;

            out << std::endl;

            return out;
        }

        std::ostream& print_node(std::ostream &out, node_t* node) {
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

int main(int argc, char** argv) {
    std::string fname_binary;
    std::string fname_map;
    std::string fname_dot;
    year_t ylength;
    double max_error;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("error", po::value(&max_error)->required(), "maximum error during tree merge")
        ("dotfile", po::value(&fname_dot), "dot file that represents the index (optional)")
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


    std::size_t depth = power_of_2(ylength);
    index_t index(depth);
    transformer trans(ylength, depth, max_error, base, idxmap, &index);

    n = 100000; // DEBUG

    std::cout << "build and merge trees" << std::endl;
    std::mt19937 rng;
    std::size_t counter = 0;
    std::vector<std::size_t> indices(n);
    std::generate(indices.begin(), indices.end(), [&counter](){
        return counter++;
    });
    std::shuffle(indices.begin(), indices.end(), rng);
    for (std::size_t i = 0; i < n; ++i) {
        if (i % 10000 == 0) {
            std::cout << "  " << i << "/" << n << std::endl;
        }
        trans.run(indices[i]);
    }
    // XXX: unshuffle superroots!
    std::cout << "done" << std::endl;

    std::cout << "stats:" << std::endl
        << "  #nodes           = " << index.node_count << std::endl
        << "  compression rate = " << (static_cast<double>(index.node_count) / static_cast<double>((ylength - 1) * n)) << std::endl;

    if (vm.count("dotfile")) {
        std::ofstream out(fname_dot);
        printer pr;
        pr.print_begin(std::cout);
        for (auto sr : index.superroots) {
            pr.print_tree(std::cout, sr);
        }
        pr.print_end(std::cout);
    }

    // free memory for sanity checking
    index.delete_all_ptrs();
}
