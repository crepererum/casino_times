#include <cassert>

#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
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
        using neighbors_t = std::vector<std::pair<node_ptr_t, inexact_t>>;

        range_bucket_t() = default;

        void insert(node_ptr_t node) {
            const slot_entry_t entry{node->x, node};
            auto lower = find_lower(entry);
            _slot.insert(lower, entry);
        }

        void get_nearest(node_ptr_t node, inexact_t max_dist, std::size_t max_size, neighbors_t& neighbors) const {
            auto begin  = _slot.cbegin();
            auto end    = _slot.cend();
            const slot_entry_t entry{node->x, node};
            auto center = find_lower(entry);

            neighbors.clear();
            auto it_up          = center;
            auto it_down        = std::prev(center);
            inexact_t dist_up   = std::numeric_limits<inexact_t>::infinity();
            inexact_t dist_down = std::numeric_limits<inexact_t>::infinity();
            auto prev_begin     = std::prev(begin); // let's also hope that std::prev(begin) works
            if (it_up != end) {
                dist_up = std::abs(it_up->first - entry.first);
            }
            if (it_down != prev_begin) {
                dist_down = std::abs(it_down->first - entry.first);
            }
            while ((neighbors.size() < max_size)
                    && (((it_up != end) && (dist_up <= max_dist)) || ((it_down != prev_begin) && (dist_down <= max_dist)))) {
                if (dist_up < dist_down) {
                    neighbors.push_back(std::make_pair(it_up->second, dist_up));
                    ++it_up;
                    if (it_up != end) {
                        dist_up = std::abs(it_up->first - entry.first);
                    } else {
                        dist_up = std::numeric_limits<inexact_t>::infinity();
                    }
                } else {
                    neighbors.push_back(std::make_pair(it_down->second, dist_down));
                    --it_down;
                    if (it_down != prev_begin) {
                        dist_down = std::abs(it_down->first - entry.first);
                    } else {
                        dist_down = std::numeric_limits<inexact_t>::infinity();
                    }
                }
            }
        }

        void delete_all_ptrs(mapped_file_ptr_t& f) {
            allocator_node_t alloc(f->get_segment_manager());
            for (auto& entry : _slot) {
                dealloc_in_mapped_file(alloc, entry.second);
            }
        }

    private:
        using slot_entry_t = std::pair<inexact_t, node_ptr_t>;
        std::vector<slot_entry_t> _slot;

        decltype(_slot)::const_iterator find_lower(const slot_entry_t& entry) const {
            return std::lower_bound(_slot.cbegin(), _slot.cend(), entry, [](const slot_entry_t& a, const slot_entry_t& b){
                return a.first < b.first;
            });
        }
};

struct index_tmp_t {
    std::vector<range_bucket_t> lowest_level;
    std::vector<std::unordered_map<children_t, range_bucket_t>> higher_levels;
    std::vector<std::size_t> node_counts;

    index_tmp_t(std::size_t depth)
        : lowest_level(std::size_t{1} << (depth - 1)),
        higher_levels(depth - 1),
        node_counts(depth, 0) {}

    range_bucket_t& find_bucket(std::size_t l, std::size_t idx, node_ptr_t current_node) {
        if (l >= higher_levels.size()) {
            return lowest_level[idx];
        } else {
            return higher_levels[l][current_node->children];
        }
    }

    void delete_all_ptrs(mapped_file_ptr_t& f) {
        allocator_superroot_t alloc(f->get_segment_manager());

        std::size_t find_count;
        superroot_vector_t* superroots;
        std::tie(superroots, find_count) = f->find<superroot_vector_t>("superroots");
        if (find_count != 1) {
            return;
        }

        for (auto& sr : *superroots) {
            if (sr != nullptr) {
                dealloc_in_mapped_file(alloc, sr);
            }
        }
        for (auto& bucket : lowest_level) {
            bucket.delete_all_ptrs(f);
        }
        for (auto& map : higher_levels) {
            for (auto& kv : map) {
                kv.second.delete_all_ptrs(f);
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

class error_calculator {
    public:
        error_calculator(std::size_t ylength, std::size_t depth, const calc_t* base, const std::shared_ptr<transformer>& transf) :
            _ylength(ylength),
            _depth(depth),
            _base(base),
            _transformer(transf),
            _delta(_ylength, 0.0),
            _delta_copy(_ylength, 0.0),
            _is_approx(false) {}

        inexact_t recalc(const std::size_t i) {
            // do idwt
            std::vector<calc_t> data_approximated(_ylength);
            _transformer->tree_to_data(data_approximated.data());

            // compare
            const calc_t* local_base = _base + (i * _ylength);
            std::get<1>(_delta) = -std::numeric_limits<inexact_t>::infinity();
            for (std::size_t y = 0; y < _ylength; ++y) {
                std::get<0>(_delta)[y] = std::abs(static_cast<inexact_t>(data_approximated[y] - local_base[y]));
                std::get<1>(_delta) = std::max(std::get<1>(_delta), std::get<0>(_delta)[y]);
            }

            // calc exact error
            inexact_t error = error_from_delta(_delta);

            _transformer->superroot->error = error;
            _is_approx = false;

            return error;
        }

        inexact_t update(std::size_t l, std::size_t idx, inexact_t dist) {
            guess_delta_update(_delta, l, idx, dist);
            inexact_t error = error_from_delta(_delta);
            _transformer->superroot->error = error;
            _is_approx = true;

            return error;
        }

        inexact_t guess_error(std::size_t l, std::size_t idx, inexact_t dist) {
            _delta_copy = _delta;
            guess_delta_update(_delta_copy, l, idx, dist);
            return error_from_delta(_delta_copy);
        }

        bool is_approx() const {
            return _is_approx;
        }

        bool is_in_range(std::size_t i, std::size_t l, std::size_t idx, inexact_t dist, inexact_t max_error) {
            bool in_range = false;

            if (is_in_range_impl(l, idx, dist, max_error)) {
                in_range = true;
            } else if (_is_approx) {
                // ok, error is an approximation anyway, so lets calculate the right value and try again
                recalc(i);

                if (is_in_range_impl(l, idx, dist, max_error)) {
                    in_range = true;
                }
            }

            return in_range;
        }

    private:
        using delta_t = std::pair<std::vector<inexact_t>, inexact_t>;

        const std::size_t            _ylength;
        const std::size_t            _depth;
        const calc_t*                _base;
        std::shared_ptr<transformer> _transformer;
        delta_t                      _delta;
        delta_t                      _delta_copy;
        bool                         _is_approx;

        inexact_t error_from_delta(const delta_t& delta) const {
            return std::get<1>(delta);
        }

        void guess_delta_update(delta_t& delta, std::size_t l, std::size_t idx, inexact_t dist) {
            std::size_t influence = 1u << (_depth - l);
            std::size_t shift     = influence * idx;
            inexact_t dist_recalced  = dist / static_cast<inexact_t>(influence);

            for (std::size_t y = shift; y < shift + influence; ++y) {
                std::get<0>(delta)[y] += dist_recalced;
                std::get<1>(delta) = std::max(std::get<1>(delta), std::get<0>(delta)[y]);
            }
        }

        bool is_in_range_impl(std::size_t l, std::size_t idx, inexact_t dist, inexact_t max_error) {
            inexact_t total       = guess_error(l, idx, dist);
            inexact_t limit_total = max_error;

            return total < limit_total;
        }
};

class engine {
    public:
        engine(std::size_t ylength, std::size_t depth, inexact_t max_error, const calc_t* base, index_tmp_t* index_tmp, index_stored_t* index_stored, const mapped_file_ptr_t& mapped_file)
            : _ylength(ylength),
            _depth(depth),
            _max_error(max_error),
            _base(base),
            _index_tmp(index_tmp),
            _index_stored(index_stored),
            _mapped_file(mapped_file),
            _alloc_superroot(_mapped_file->get_segment_manager()),
            _alloc_node(_mapped_file->get_segment_manager()),
            _transformer(std::make_shared<transformer>(_ylength, _depth)),
            _error_calc(std::make_shared<error_calculator>(_ylength, _depth, _base, _transformer)) {}

        superroot_ptr_t run(std::size_t i) {
            _transformer->data_to_tree(_base + (i * _ylength));
            _error_calc->recalc(i);  // correct error because of floating point errors
            run_mergeloop(i);
            drain(i);
            _error_calc->recalc(i);  // correct error one last time

            // move superroot to mapped file
            (*_index_stored->superroots)[i] = alloc_in_mapped_file(_alloc_superroot);
            *((*_index_stored->superroots)[i]) = *(_transformer->superroot);

            return _transformer->superroot;
        }

    private:
        struct queue_entry_t {
            inexact_t   score;
            inexact_t   dist;
            std::size_t l;
            std::size_t idx;
            node_ptr_t  neighbor;

            queue_entry_t() = default;

            queue_entry_t(inexact_t score_, inexact_t dist_, std::size_t l_, std::size_t idx_, node_ptr_t neighbor_) :
                score(score_),
                dist(dist_),
                l(l_),
                idx(idx_),
                neighbor(neighbor_) {}
        };

        struct queue_entry_compare {
            bool operator()(const queue_entry_t& a, const queue_entry_t& b) {
                return a.score > b.score;
            }
        };

        using queue_t = std::priority_queue<queue_entry_t, std::vector<queue_entry_t>, queue_entry_compare>;

        const std::size_t                 _ylength;
        const std::size_t                 _depth;
        inexact_t                         _max_error;
        const calc_t*                     _base;
        index_tmp_t*                      _index_tmp;
        index_stored_t*                   _index_stored;
        std::mt19937                      _rng;
        mapped_file_ptr_t                 _mapped_file;
        allocator_superroot_t             _alloc_superroot;
        allocator_node_t                  _alloc_node;
        std::shared_ptr<transformer>      _transformer;
        std::shared_ptr<error_calculator> _error_calc;
        queue_t                           _queue;
        range_bucket_t::neighbors_t       _neighbors;

        void run_mergeloop(std::size_t i) {
            assert(queue.empty());

            // fill queue with entries from lowest level
            std::vector<std::size_t> indices(_transformer->levels[_depth - 1].size());
            for (std::size_t idx = 0; idx < indices.size(); ++idx) {
                indices[idx] = idx;
            }
            std::shuffle(indices.begin(), indices.end(), _rng);
            for (auto idx : indices) {
                std::size_t l = _depth - 1;
                generate_queue_entries(l, idx, _queue);
            }

            // now run merge loop
            while (!_queue.empty()) {
                auto best = _queue.top();
                _queue.pop();

                node_ptr_t current_node = _transformer->levels[best.l][best.idx];

                // check if node was already merged and if distance is low enough
                if (current_node != nullptr) {
                    // check if merge would be in error range (i.e. does not increase error beyond current_max_error)
                    if (_error_calc->is_in_range(i, best.l, best.idx, best.dist, _max_error)) {
                        // calculate approx. error
                        _error_calc->update(best.l, best.idx, best.dist);

                        // execute merge
                        _transformer->link_to_parent(best.neighbor, best.l, best.idx);

                        // remove old node and mark as merged
                        _transformer->levels[best.l][best.idx] = nullptr;

                        // generate new queue entries
                        std::size_t idx_even = best.idx - (best.idx % 2);
                        if ((best.l > 0) && (_transformer->levels[best.l][idx_even] == nullptr) && (_transformer->levels[best.l][idx_even + 1] == nullptr)) {
                            generate_queue_entries(best.l - 1, best.idx >> 1, _queue);
                        }
                    }
                }
            }
        }

        void generate_queue_entries(std::size_t l, std::size_t idx, queue_t& queue) {
            node_ptr_t current_node = _transformer->levels[l][idx];

            if (current_node != nullptr) {
                auto& bucket = _index_tmp->find_bucket(l, idx, current_node);
                bucket.get_nearest(current_node, std::numeric_limits<inexact_t>::infinity(), 1, _neighbors);
                if (!_neighbors.empty()) {
                    inexact_t error = _error_calc->guess_error(l, idx, _neighbors[0].second);
                    inexact_t score = error - _transformer->superroot->error;
                    queue.emplace(score, _neighbors[0].second, l, idx, _neighbors[0].first);
                }
            }
        }

        /*
         * adds all remaining to index, without any merges
         */
        void drain(std::size_t i) {
            // do work bottom to top, otherwise node relinking ain't working
            for (std::size_t l_plus = _depth; l_plus > 0; --l_plus) {
                std::size_t l = l_plus - 1;

                for (std::size_t idx = 0; idx < _transformer->levels[l].size(); ++idx) {
                    node_ptr_t current_node = _transformer->levels[l][idx];

                    // node might already be merged
                    if (current_node != nullptr) {
                        // recreate node within the mapped file
                        auto node_stored = alloc_in_mapped_file(_alloc_node);
                        *node_stored = *current_node;
                        _transformer->link_to_parent(node_stored, l, idx);

                        // register a new parent for the child nodes
                        if (l == 0) {
                            _index_stored->register_superroot(node_stored, (*(_index_stored->superroots))[i]);
                        }
                        for (const auto& c : node_stored->children) {
                            if (c) {
                                _index_stored->register_parent(c, node_stored);
                            }
                        }

                        // add node to value lookup system
                        auto& bucket = _index_tmp->find_bucket(l, idx, node_stored);
                        bucket.insert(node_stored);
                        ++_index_tmp->node_counts[l];
                    }
                }
            }
        }
};

double calc_compression_rate(const index_tmp_t* index, year_t ylength, std::size_t n) {
    std::size_t size_normal = sizeof(calc_t) * static_cast<std::size_t>(ylength) * n;
    std::size_t size_compression = sizeof(node_t) * index->total_node_count() + sizeof(superroot_t) * n;

    return static_cast<double>(size_compression) / static_cast<double>(size_normal);
}

void print_process(const index_tmp_t* index, year_t ylength, std::size_t n, std::size_t i) {
    std::cout << "  process=" << i << "/" << n
        << " #nodes=" << index->total_node_count()
        << " %nodes=" << (static_cast<double>(index->total_node_count()) / static_cast<double>((ylength - 1) * i))
        << " compression_rate=" << calc_compression_rate(index, ylength, i)
        << std::endl;
}

void print_stats(const index_tmp_t* index, std::size_t n) {
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

void trace_up(const std::vector<node_ptr_t>& nodes, const index_stored_t* index, std::unordered_set<superroot_ptr_t>& srs) {
    std::vector<node_ptr_t> next;

    for (const auto& n : nodes) {
        auto parents = index->find_parents(n);
        if (parents) {
            for (const auto& p : *parents) {
                next.push_back(p);
            }
        }

        auto superroots = index->find_superroots(n);
        if (superroots) {
            for (const auto& s : *superroots) {
                srs.insert(s);
            }
        }
    }

    if (!next.empty()) {
        trace_up(next, index, srs);
    }
}

void trace_down(const std::vector<node_ptr_t>& nodes, const index_stored_t* index) {
    std::unordered_set<superroot_ptr_t> srs_set;
    std::vector<node_ptr_t> next;

    std::cout << "  #nodes=" << nodes.size() << " :" << std::flush;

    for (const auto& n : nodes) {
        std::unordered_set<superroot_ptr_t> srs_subset;
        trace_up({n}, index, srs_subset);
        for (const auto& s : srs_subset) {
            srs_set.insert(s);
        }

        if (srs_subset.size() == 1) {
            std::cout << " .";
        } else {
            std::cout << " " << (srs_subset.size() - 1);
        }
        std::cout << std::flush;

        for (const auto& c : n->children) {
            if (c != nullptr) {
                next.push_back(c);
            }
        }
    }

    if (srs_set.size() == 1) {
        std::cout << " = ." << std::endl;

    } else {
        std::cout << " = " << (srs_set.size() - 1) << std::endl;
    }

    if (!next.empty()) {
        trace_down(next, index);
    }
}

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_binary;
    std::string fname_map;
    std::string fname_index;
    std::size_t index_size;
    year_t ylength;
    inexact_t max_error;
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
    std::cout << "done" << std::endl;

    std::cout << "build and merge trees" << std::endl;
    std::size_t depth = power_of_2(ylength);
    index_tmp_t index_tmp(depth);
    index_stored_t index_stored(findex, n);
    engine eng(ylength, depth, max_error, base, &index_tmp, &index_stored, findex);
    std::mt19937 rng;
    std::vector<std::pair<std::size_t, std::size_t>> indices;
    constexpr std::size_t blocksize = 16;
    for (std::size_t i = 0; i < n; i += blocksize) {
        std::size_t begin = i;
        std::size_t end = std::min(begin + blocksize, n);
        indices.emplace_back(std::make_pair(begin, end));
    }
    std::shuffle(indices.begin(), indices.end(), rng);

    std::size_t counter = 0;
    for (std::size_t b = 0; b < indices.size(); ++b) {
        if (b % 50 == 0) {
            print_process(&index_tmp, ylength, n, counter);
        }

        const auto& block = indices[b];
        std::size_t begin = block.first;
        std::size_t end = block.second;

        for (std::size_t i = begin; i < end; ++i) {
            __builtin_prefetch(&base[(i + 1) * ylength], 0, 0);
            eng.run(i);
        }

        counter += (end - begin);
    }
    print_process(&index_tmp, ylength, n, n);
    std::cout << "done" << std::endl;

    std::cout << "trace one path:" << std::endl;
    trace_down({(*index_stored.superroots)[indices[indices.size() - 1].first]->root}, &index_stored);  // XXX: find a better example trace

    // free memory for sanity checking
    // DONT! only for debugging!
    //index.delete_all_ptrs(findex);

    std::cout << "Free memory: " << (findex->get_free_memory() >> 10) << "k of " << (findex->get_size() >> 10) << "k" << std::endl;

    print_stats(&index_tmp, n);
}
