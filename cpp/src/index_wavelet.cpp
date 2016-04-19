#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"
#include "wavelet_transformer.hpp"
#include "wavelet_tree.hpp"

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
    std::vector<std::size_t> node_counts;

    index_t(superroot_vector_t* superroots_, std::size_t depth) : superroots(superroots_), levels(depth), node_counts(depth, 0) {
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

    std::size_t total_node_count() const {
        return std::accumulate(
            node_counts.begin(),
            node_counts.end(),
            static_cast<std::size_t>(0),
            std::plus<std::size_t>()
        );
    }
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
                        ++_index->node_counts[l];
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

double calc_compression_rate(const index_t* index, year_t ylength, std::size_t n) {
    std::size_t size_normal = sizeof(calc_t) * static_cast<std::size_t>(ylength) * n;
    std::size_t size_compression = sizeof(node_t) * index->total_node_count() + sizeof(superroot_t) * n;

    return static_cast<double>(size_compression) / static_cast<double>(size_normal);
}

void print_process(const index_t* index, year_t ylength, std::size_t n, std::size_t i) {
    std::cout << "  process=" << i << "/" << n
        << " #nodes=" << index->total_node_count()
        << " %nodes=" << (static_cast<double>(index->total_node_count()) / static_cast<double>((ylength - 1) * i))
        << " compression_rate=" << calc_compression_rate(index, ylength, i)
        << std::endl;
}

void print_stats(const index_t* index, std::size_t n) {
    std::size_t sum_is = 0;
    std::size_t sum_should = 0;

    std::cout << "stats:" << std::endl;
    for (std::size_t l = 0; l < index->node_counts.size(); ++l) {
        std::size_t is = index->node_counts[l];
        std::size_t should = (static_cast<std::size_t>(1) << l) * n;
        double relative = static_cast<double>(is) / static_cast<double>(should);

        std::cout << "  level=" << l << " #nodes=" << is << " %nodes=" << relative << std::endl;

        sum_is += is;
        sum_should += should;
    }

    double sum_relative = static_cast<double>(sum_is) / static_cast<double>(sum_should);
    std::cout << "  level=X #nodes=" << sum_is << " %nodes=" << sum_relative << std::endl;
}

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_binary;
    std::string fname_map;
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
        ("index", po::value(&fname_index)->required(), "index file")
        ("size", po::value(&index_size)->required(), "size of the index file (in bytes)")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "index_wavelet")) {
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

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
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

    for (std::size_t i = 0; i < n; ++i) {
        if (i % 1000 == 0) {
            print_process(&index, ylength, n, i);
        }
        eng.run(indices[i]);
    }
    print_process(&index, ylength, n, n);
    std::cout << "done" << std::endl;

    // free memory for sanity checking
    // DONT! only for debugging!
    //index.delete_all_ptrs(findex);

    std::cout << "Free memory: " << (findex->get_free_memory() >> 10) << "k of " << (findex->get_size() >> 10) << "k" << std::endl;

    print_stats(&index, n);
}
