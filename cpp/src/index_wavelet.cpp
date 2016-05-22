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

        std::vector<std::pair<node_ptr_t, inexact_t>> get_nearest(node_ptr_t node, inexact_t max_dist, std::size_t max_size) {
            auto begin  = _slot.begin();
            auto end    = _slot.end();
            auto center = std::lower_bound(begin, end, node, [](node_ptr_t a, node_ptr_t b) {
                return a->x < b->x;
            });

            std::vector<std::pair<node_ptr_t, inexact_t>> neighbors;
            auto it_up          = center;
            auto it_down        = std::prev(center);
            inexact_t dist_up   = std::numeric_limits<inexact_t>::infinity();
            inexact_t dist_down = std::numeric_limits<inexact_t>::infinity();
            auto prev_begin     = std::prev(begin); // let's also hope that std::prev(begin) works
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
                        dist_up = std::numeric_limits<inexact_t>::infinity();
                    }
                } else {
                    neighbors.push_back(std::make_pair(*it_down, dist_down));
                    --it_down;
                    if (it_down != prev_begin) {
                        dist_down = std::abs((*it_down)->x - node->x);
                    } else {
                        dist_down = std::numeric_limits<inexact_t>::infinity();
                    }
                }
            }

            return neighbors;
        }

        void delete_all_ptrs(mapped_file_ptr_t& f) {
            allocator_node_t alloc(f->get_segment_manager());
            for (auto& node : _slot) {
                dealloc_in_mapped_file(alloc, node);
            }
        }

    private:
        std::vector<node_ptr_t> _slot;
};

using rbucket_t = std::unique_ptr<range_bucket_t>;

struct index_t {
    superroot_vector_t* superroots;
    std::vector<rbucket_t> lowest_level;
    std::vector<std::unordered_map<children_t, rbucket_t>> higher_levels;
    std::vector<std::size_t> node_counts;

    index_t(superroot_vector_t* superroots_, std::size_t depth)
        : superroots(superroots_),
        lowest_level(1 << (depth - 1)),
        higher_levels(depth - 1),
        node_counts(depth, 0) {}

    rbucket_t& find_bucket(std::size_t l, std::size_t idx, node_ptr_t current_node) {
        if (l >= higher_levels.size()) {
            auto& b = lowest_level[idx];
            if (!b) {
                b = std::make_unique<range_bucket_t>();
            }
            return b;
        } else {
            auto& b = higher_levels[l][current_node->children];
            if (!b) {
                b = std::make_unique<range_bucket_t>();
            }
            return b;
        }
    }

    void delete_all_ptrs(mapped_file_ptr_t& f) {
        allocator_superroot_t alloc(f->get_segment_manager());
        for (auto& sr : *superroots) {
            if (sr != nullptr) {
                dealloc_in_mapped_file(alloc, sr);
            }
        }
        for (auto& bucket : lowest_level) {
            bucket->delete_all_ptrs(f);
        }
        for (auto& map : higher_levels) {
            for (auto& kv : map) {
                kv.second->delete_all_ptrs(f);
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
        // XXX: align vector to better use AVX
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

struct queue_entry_t;
using qentry_t = std::unique_ptr<queue_entry_t>;

struct qentry_compare {
    bool operator()(const qentry_t& a, const qentry_t& b);
};

using queue_t      = std::priority_queue<qentry_t, std::vector<qentry_t>, qentry_compare>;
using generation_t = std::vector<std::vector<std::size_t>>;

struct queue_entry_t {
    inexact_t   score;
    std::size_t l;
    std::size_t idx;
    std::size_t gen;

    queue_entry_t(inexact_t score_, std::size_t l_, std::size_t idx_, std::size_t gen_) :
        score(score_),
        l(l_),
        idx(idx_),
        gen(gen_) {}
    virtual ~queue_entry_t() = default;

    void operator()(
            inexact_t max_error,
            index_t* index,
            const std::shared_ptr<transformer>& transformer,
            const std::shared_ptr<error_calculator>& error_calc,
            std::size_t i,
            queue_t& queue,
            generation_t& generation
        ) {
        node_ptr_t current_node = transformer->levels[l][idx];
        std::size_t current_gen = generation[l][idx];

        // check if node was already merged
        if ((current_node != nullptr) && (gen == current_gen)) {
            operator_impl(max_error, index, transformer, error_calc, i, current_node, queue, generation);
        }
    }

    virtual void operator_impl(
            inexact_t max_error,
            index_t* index,
            const std::shared_ptr<transformer>& transformer,
            const std::shared_ptr<error_calculator>& error_calc,
            std::size_t i,
            node_ptr_t current_node,
            queue_t& queue,
            generation_t& generation
        ) = 0;
};

bool qentry_compare::operator()(const qentry_t& a, const qentry_t& b) {
    return a->score > b->score;
}

struct queue_entry_merge_t : public queue_entry_t {
    inexact_t       dist;
    node_ptr_t      neighbor;
    range_bucket_t* bucket;

    queue_entry_merge_t(inexact_t score_, inexact_t dist_, std::size_t l_, std::size_t idx_, std::size_t gen_, node_ptr_t neighbor_, range_bucket_t* bucket_) :
        queue_entry_t(score_, l_, idx_, gen_),
        dist(dist_),
        neighbor(neighbor_),
        bucket(bucket_) {}

    virtual void operator_impl(
            inexact_t max_error,
            index_t* index,
            const std::shared_ptr<transformer>& transformer,
            const std::shared_ptr<error_calculator>& error_calc,
            std::size_t i,
            node_ptr_t current_node,
            queue_t& queue,
            generation_t& generation) override {
        // check if merge would be in error range (i.e. does not increase error beyond current_max_error)
        if (error_calc->is_in_range(i, l, idx, dist, max_error)) {
            // calculate approx. error
            error_calc->update(l, idx, dist);

            // execute merge
            transformer->link_to_parent(neighbor, l, idx);

            // remove old node and mark as merged
            delete current_node.get();
            transformer->levels[l][idx] = nullptr;

            // generate new queue entries
            std::size_t idx_even = idx - (idx % 2);
            if ((l > 0) && (transformer->levels[l][idx_even] == nullptr) && (transformer->levels[l][idx_even + 1] == nullptr)) {
                queue_entry_merge_t::generate(index, transformer, error_calc, l - 1, idx >> 1, queue, generation, nullptr);
            }
        }
    }

    static void generate(
            index_t* index,
            const std::shared_ptr<transformer>& transformer,
            const std::shared_ptr<error_calculator>& error_calc,
            std::size_t l,
            std::size_t idx,
            queue_t& queue,
            generation_t& generation,
            range_bucket_t* bucket = nullptr) {
        node_ptr_t current_node = transformer->levels[l][idx];

        if (current_node != nullptr) {
            if (bucket == nullptr) {
                bucket = (index->find_bucket(l, idx, current_node)).get();
            }
            auto neighbors = bucket->get_nearest(current_node, std::numeric_limits<inexact_t>::infinity(), 1);
            if (!neighbors.empty()) {
                inexact_t error = error_calc->guess_error(l, idx, neighbors[0].second);
                inexact_t score = error - transformer->superroot->error;
                queue.emplace(static_cast<queue_entry_t*>(
                        new queue_entry_merge_t(score, neighbors[0].second, l, idx, generation[l][idx], neighbors[0].first, bucket)
                ));
            }
        }
    }
};

struct queue_entry_prune_t : public queue_entry_t {
    static constexpr std::size_t stepsize = 2;
    static constexpr std::size_t stepmax  = 30;

    static constexpr std::uint32_t masks[] = {
        ~gen_mask( 0),
        ~gen_mask( 1),
        ~gen_mask( 2),
        ~gen_mask( 3),
        ~gen_mask( 4),
        ~gen_mask( 5),
        ~gen_mask( 6),
        ~gen_mask( 7),
        ~gen_mask( 8),
        ~gen_mask( 9),
        ~gen_mask(10),
        ~gen_mask(11),
        ~gen_mask(12),
        ~gen_mask(13),
        ~gen_mask(14),
        ~gen_mask(15),
        ~gen_mask(16),
        ~gen_mask(17),
        ~gen_mask(18),
        ~gen_mask(19),
        ~gen_mask(20),
        ~gen_mask(21),
        ~gen_mask(22),
        ~gen_mask(23),
        ~gen_mask(24),
        ~gen_mask(25),
        ~gen_mask(26),
        ~gen_mask(27),
        ~gen_mask(28),
        ~gen_mask(29),
        ~gen_mask(30),
    };
    static_assert(sizeof(masks) / sizeof(std::uint32_t) == stepmax + 1, "mask array isn't consistent!");

    inexact_t   target;
    inexact_t   dist;
    std::size_t p;

    queue_entry_prune_t(inexact_t score_, inexact_t target_, inexact_t dist_, std::size_t l_, std::size_t idx_, std::size_t gen_, std::size_t p_) :
        queue_entry_t(score_, l_, idx_, gen_),
        target(target_),
        dist(dist_),
        p(p_) {}

    virtual void operator_impl(
            inexact_t max_error,
            index_t* index,
            const std::shared_ptr<transformer>& transformer,
            const std::shared_ptr<error_calculator>& error_calc,
            std::size_t i,
            node_ptr_t current_node,
            queue_t& queue,
            generation_t& generation) override {
        if (error_calc->is_in_range(i, l, idx, dist, max_error)) {
            error_calc->update(l, idx, dist);
            current_node->x = target;

            // invalidate old merge requests
            ++generation[l][idx];

            // generate new queue entries
            queue_entry_prune_t::generate(index, transformer, error_calc, l, idx, queue, generation, p + stepsize);
            queue_entry_merge_t::generate(index, transformer, error_calc, l, idx, queue, generation);
        }
    }

    static void generate(
            index_t* index,
            const std::shared_ptr<transformer>& transformer,
            const std::shared_ptr<error_calculator>& error_calc,
            std::size_t l,
            std::size_t idx,
            queue_t& queue,
            generation_t& generation,
            std::size_t p = stepsize) {
        node_ptr_t current_node = transformer->levels[l][idx];

        if (current_node != nullptr && p <= stepmax) {
            inexact_t target = prune_value(current_node->x, p);

            if (target == current_node->x) {  // exact comparison is legal and desired here, because of the bitmask trick
                queue_entry_prune_t::generate(index, transformer, error_calc, l, idx, queue, generation, p + stepsize);
            } else {
                inexact_t dist = std::abs(current_node->x - target);
                inexact_t error = error_calc->guess_error(l, idx, dist);
                inexact_t score = error - transformer->superroot->error;
                score *= 100.00f; // XXX: good? bad? config?
                queue.emplace(static_cast<queue_entry_t*>(
                    new queue_entry_prune_t{score, target, dist, l, idx, generation[l][idx], p}
                ));
            }
        }
    }

    static inexact_t prune_value(inexact_t x, std::size_t n) {
        union {
            inexact_t f;
            std::uint32_t i;
        } u;
        static_assert(sizeof(inexact_t) == sizeof(std::uint32_t), "that mask trick isn't working!");
        u.f = x;
        u.i &= masks[n];
        return u.f;
    }
};
constexpr std::uint32_t queue_entry_prune_t::masks[];  // C++ is weird

class engine {
    public:
        engine(std::size_t ylength, std::size_t depth, inexact_t max_error, const calc_t* base, index_t* index, const mapped_file_ptr_t& mapped_file)
            : _ylength(ylength),
            _depth(depth),
            _max_error(max_error),
            _base(base),
            _index(index),
            _mapped_file(mapped_file),
            _alloc_superroot(_mapped_file->get_segment_manager()),
            _alloc_node(_mapped_file->get_segment_manager()),
            _transformer(std::make_shared<transformer>(_ylength, _depth)),
            _error_calc(std::make_shared<error_calculator>(_ylength, _depth, _base, _transformer)) {}

        superroot_ptr_t run(std::size_t i) {
            _transformer->data_to_tree(_base + (i * _ylength));
            _error_calc->recalc(i);  // correct error because of floating point errors
            run_mergeloop(i);
            drain();
            _error_calc->recalc(i);  // correct error one last time

            // move superroot to mapped file
            (*_index->superroots)[i] = alloc_in_mapped_file(_alloc_superroot);
            *((*_index->superroots)[i]) = *(_transformer->superroot);
            delete _transformer->superroot.get();

            return _transformer->superroot;
        }

    private:
        const std::size_t                 _ylength;
        const std::size_t                 _depth;
        inexact_t                         _max_error;
        const calc_t*                     _base;
        index_t*                          _index;
        std::mt19937                      _rng;
        mapped_file_ptr_t                 _mapped_file;
        allocator_superroot_t             _alloc_superroot;
        allocator_node_t                  _alloc_node;
        std::shared_ptr<transformer>      _transformer;
        std::shared_ptr<error_calculator> _error_calc;
        queue_t                           _queue;

        void run_mergeloop(std::size_t i) {
            // clear queue
            while (!_queue.empty()) {
                _queue.pop();
            }

            // build generation counter
            generation_t generation(_depth);
            for (std::size_t l = 0; l < _depth; ++l) {
                generation[l] = std::vector<std::size_t>(1u << l, 0);
            }

            // fill queue with entries from lowest level
            std::vector<std::pair<std::size_t, std::size_t>> indices;
            for (std::size_t l_plus = _depth; l_plus > 0; --l_plus) {
                std::size_t l = l_plus - 1;

                for (std::size_t idx = 0; idx < _transformer->levels[l].size(); ++idx) {
                    indices.push_back(std::make_pair(l, idx));
                }
            }
            std::shuffle(indices.begin(), indices.end(), _rng);
            for (auto p : indices) {
                std::size_t l = p.first;
                std::size_t idx = p.second;
                queue_entry_merge_t::generate(_index, _transformer, _error_calc, l, idx, _queue, generation);
                queue_entry_prune_t::generate(_index, _transformer, _error_calc, l, idx, _queue, generation);
            }

            // now run merge loop
            while (!_queue.empty()) {
                qentry_t best = std::move(const_cast<qentry_t&>(_queue.top()));
                _queue.pop();

                (*best)(_max_error, _index, _transformer, _error_calc, i, _queue, generation);
            }
        }

        /*
         * adds all remaining to index, without any merges
         */
        void drain() {
            // do work bottom to top, otherwise node relinking ain't working
            for (std::size_t l_plus = _depth; l_plus > 0; --l_plus) {
                std::size_t l = l_plus - 1;

                for (std::size_t idx = 0; idx < _transformer->levels[l].size(); ++idx) {
                    node_ptr_t current_node = _transformer->levels[l][idx];

                    // node might already be merged
                    if (current_node != nullptr) {
                        auto node_stored = alloc_in_mapped_file(_alloc_node);
                        *node_stored = *current_node;
                        _transformer->link_to_parent(node_stored, l, idx);
                        delete current_node.get();

                        auto& bucket = _index->find_bucket(l, idx, node_stored);
                        bucket->insert(node_stored);
                        ++_index->node_counts[l];
                    }
                }
            }
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
    auto segment_manager = findex->get_segment_manager();
    allocator_superroot_ptr_t allocator_superroot(segment_manager);
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

    n = 100000;  // DEBUG
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
