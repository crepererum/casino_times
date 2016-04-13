#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include <flann/flann.hpp>

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

    template<>
    struct hash<std::tuple<node_t*, node_t*, node_t*, node_t*>> {
        size_t operator()(const std::tuple<node_t*, node_t*, node_t*, node_t*>& obj) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, std::get<0>(obj));
            boost::hash_combine(seed, std::get<1>(obj));
            boost::hash_combine(seed, std::get<2>(obj));
            boost::hash_combine(seed, std::get<3>(obj));
            return seed;
        }
    };

    template<std::size_t N>
    struct hash<std::array<node_t*, N>> {
        size_t operator()(const std::array<node_t*, N>& obj) const {
            std::size_t seed = 0;
            for (std::size_t i = 0; i < N; ++i) {
                boost::hash_combine(seed, obj[i]);
            }
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

        std::vector<std::pair<node_t*, double>> get_nearest(node_t* node, double max_dist, std::size_t max_size) {
            auto begin  = _slot.begin();
            auto end    = _slot.end();
            auto center = std::lower_bound(begin, end, node, [](node_t* a, node_t* b) {
                return a->x < b->x;
            });

            std::vector<std::pair<node_t*, double>> neighbors;
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

        void delete_all_ptrs() {
            for (auto& node : _slot) {
                delete node;
            }
        }

    private:
        std::vector<node_t*> _slot;
};

class range_bucket2_t {
    public:
        range_bucket2_t() = default;

        void insert(node_t* node) {
            if (node->child_l != nullptr && node->child_r != nullptr) {
                std::unique_ptr<double[]> data(new double[3]);
                data[0] = node->x;
                data[1] = node->child_l->x;
                data[2] = node->child_r->x;

                flann::Matrix<double> matrix(data.get(), 1, 3);

                if (_flann) {
                    _flann->addPoints(matrix);
                } else {
                    _flann = std::make_unique<flann::Index<flann::L1<double>>>(matrix, flann::KDTreeSingleIndexParams());
                    _flann->buildIndex();
                }
                _flann_helper.push_back(node);
            }
        }

        std::vector<std::pair<node_t*, double>> get_nearest(node_t* node, double max_dist, std::size_t max_size) {
            if (_flann) {
                std::vector<int>    data_indices(max_size, -1);
                std::vector<double> data_dist(max_size, std::numeric_limits<double>::infinity());
                std::vector<double> data_pos(3);

                data_pos[0] = node->x;
                data_pos[1] = node->child_l->x;
                data_pos[2] = node->child_r->x;

                flann::Matrix<int> indices(data_indices.data(), 1, max_size);
                flann::Matrix<double> dists(data_dist.data(), 1, max_size);
                flann::Matrix<double> pos(data_pos.data(), 1, 3);

                // do a knn search, using 128 checks
                _flann->knnSearch(pos, indices, dists, max_size, flann::SearchParams(128));

                // check results
                std::vector<std::pair<node_t*, double>> result;
                for (std::size_t i = 0; i < max_size; ++i) {
                    int idx = data_indices[i];
                    if (idx >= 0) {
                        node_t* other = _flann_helper[static_cast<std::size_t>(idx)];
                        double dist = std::abs(other->x - data_pos[0]) + std::abs(other->child_l->x - data_pos[1]) + std::abs(other->child_r->x - data_pos[2]);
                        if (dist < max_dist) {
                            result.push_back(std::make_pair(other, dist));
                        }
                    }
                }

                std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

                return result;
            } else {
                return {};
            }
        }

    private:
        std::unique_ptr<flann::Index<flann::L1<double>>> _flann;
        std::vector<node_t*> _flann_helper;
};

class range_bucket4_t {
    public:
        range_bucket4_t() = default;

        void insert(node_t* node) {
            if (node->child_l != nullptr && node->child_r != nullptr && node->child_l->child_l != nullptr && node->child_l->child_r != nullptr && node->child_r->child_l != nullptr && node->child_r->child_r != nullptr) {
                std::unique_ptr<double[]> data(new double[7]);
                data[0] = node->x;
                data[1] = node->child_l->x;
                data[2] = node->child_r->x;
                data[3] = node->child_l->child_l->x;
                data[4] = node->child_l->child_r->x;
                data[5] = node->child_r->child_l->x;
                data[6] = node->child_r->child_r->x;

                flann::Matrix<double> matrix(data.get(), 1, 7);

                if (_flann) {
                    _flann->addPoints(matrix);
                } else {
                    _flann = std::make_unique<flann::Index<flann::L1<double>>>(matrix, flann::KDTreeSingleIndexParams());
                    _flann->buildIndex();
                }
                _flann_helper.push_back(node);
            }
        }

        std::vector<std::pair<node_t*, double>> get_nearest(node_t* node, double max_dist, std::size_t max_size) {
            if (_flann) {
                std::vector<int>    data_indices(max_size, -1);
                std::vector<double> data_dist(max_size, std::numeric_limits<double>::infinity());
                std::vector<double> data_pos(7);

                data_pos[0] = node->x;
                data_pos[1] = node->child_l->x;
                data_pos[2] = node->child_r->x;
                data_pos[3] = node->child_l->child_l->x;
                data_pos[4] = node->child_l->child_r->x;
                data_pos[5] = node->child_r->child_l->x;
                data_pos[6] = node->child_r->child_r->x;

                flann::Matrix<int> indices(data_indices.data(), 1, max_size);
                flann::Matrix<double> dists(data_dist.data(), 1, max_size);
                flann::Matrix<double> pos(data_pos.data(), 1, 7);

                // do a knn search, using 128 checks
                _flann->knnSearch(pos, indices, dists, max_size, flann::SearchParams(128));

                // check results
                std::vector<std::pair<node_t*, double>> result;
                for (std::size_t i = 0; i < max_size; ++i) {
                    int idx = data_indices[i];
                    if (idx >= 0) {
                        node_t* other = _flann_helper[static_cast<std::size_t>(idx)];
                        double dist = std::abs(other->x - data_pos[0])
                            + std::abs(other->child_l->x - data_pos[1])
                            + std::abs(other->child_r->x - data_pos[2])
                            + std::abs(other->child_l->child_l->x - data_pos[3])
                            + std::abs(other->child_l->child_r->x - data_pos[4])
                            + std::abs(other->child_r->child_l->x - data_pos[5])
                            + std::abs(other->child_r->child_r->x - data_pos[6]);
                        if (dist < max_dist) {
                            result.push_back(std::make_pair(other, dist));
                        }
                    }
                }

                std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

                return result;
            } else {
                return {};
            }
        }

    private:
        std::unique_ptr<flann::Index<flann::L1<double>>> _flann;
        std::vector<node_t*> _flann_helper;
};

class range_index_t {
    public:
        range_index_t() = default;

        range_bucket_t& get_bucket(const node_t* node) {
            return _index_struct[std::make_pair(node->child_l, node->child_r)];
        }

        range_bucket2_t& get_bucket2(const node_t* node) {
            return _index_struct2[std::make_tuple(node->child_l->child_l, node->child_l->child_r, node->child_r->child_l, node->child_r->child_r)];
        }

        range_bucket4_t& get_bucket4(const node_t* node) {
            return _index_struct4[{{
                node->child_l->child_l->child_l,
                node->child_l->child_l->child_r,
                node->child_l->child_r->child_l,
                node->child_l->child_r->child_r,
                node->child_r->child_l->child_l,
                node->child_r->child_l->child_r,
                node->child_r->child_r->child_l,
                node->child_r->child_r->child_r
            }}];
        }

        void delete_all_ptrs() {
            for (auto& kv : _index_struct) {
                kv.second.delete_all_ptrs();
            }
        }

    private:
        // XXX: make node pointers in key const
        std::unordered_map<std::pair<node_t*, node_t*>, range_bucket_t>                     _index_struct;
        std::unordered_map<std::tuple<node_t*, node_t*, node_t*, node_t*>, range_bucket2_t> _index_struct2;
        std::unordered_map<std::array<node_t*, 8>, range_bucket4_t>                         _index_struct4;
};

struct index_t {
    std::vector<superroot_t*> superroots;
    std::vector<std::vector<range_index_t>> levels;
    std::size_t node_count;

    index_t(std::size_t depth, std::size_t n) : superroots(n, nullptr), levels(depth), node_count(0) {
        for (std::size_t l = 0; l < depth; ++l) {
            levels[l] = std::vector<range_index_t>(1 << l);
        }
    }

    void delete_all_ptrs() {
        for (auto& sr : superroots) {
            if (sr != nullptr) {
                delete sr;
            }
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
        superroot_t*                      superroot;
        std::vector<std::vector<node_t*>> levels;

        transformer(std::size_t ylength, std::size_t depth) :
            superroot(nullptr),
            levels(depth),
            _ylength(ylength),
            _depth(depth),
            _mywt(std::make_shared<wavelet>("haar"), "dwt", static_cast<int>(_ylength), static_cast<int>(_depth)),
            _influence_sqrt(_depth)
        {
            _mywt.extension("per");
            _mywt.conv("direct");

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t influence = 1u << (_depth - l);
                _influence_sqrt[l] = std::sqrt(static_cast<double>(influence));
            }
        }

        superroot_t* data_to_tree(const calc_t* data, const ngram_t& ngram) {
            _mywt.run_dwt(data);

            superroot = new superroot_t;
            superroot->ngram  = ngram;
            superroot->approx = _mywt.output()[0];
            superroot->error  = 0;

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t outdelta  = 1 << l;
                std::size_t width     = outdelta;  // same calculation

                levels[l].clear();

                for (std::size_t idx = 0; idx < width; ++idx) {
                    node_t* node  = new node_t;
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
            std::vector<node_t*> layer_a{superroot->root};
            std::vector<node_t*> layer_b;
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
        void link_to_parent(node_t* node, std::size_t l, std::size_t idx) {
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
};

class engine {
    public:
        engine(std::size_t ylength, std::size_t depth, double max_error, const calc_t* base, const std::shared_ptr<idx_ngram_map_t>& idxmap, index_t* index)
            : _ylength(ylength),
            _depth(depth),
            _max_error(max_error),
            _base(base),
            _idxmap(idxmap),
            _index(index),
            _transformer(_ylength, _depth) {}

        superroot_t* run(std::size_t i) {
            _transformer.data_to_tree(_base + (i * _ylength), (*_idxmap)[i]);
            _transformer.superroot->error = calc_error(i);  // correct error because of floating point errors
            _index->superroots[i] = _transformer.superroot;
            run_mergeloop(i);
            drain();
            _transformer.superroot->error = calc_error(i);  // correct error one last time

            return _transformer.superroot;
        }

    private:
        struct merge_entry_t {
            std::size_t l;
            std::size_t idx;
            node_t*     neighbor;

            merge_entry_t(std::size_t l_, std::size_t idx_, node_t* neighbor_) :
                l(l_),
                idx(idx_),
                neighbor(neighbor_) {}
        };

        struct queue_entry_t {
            double costs;
            double sum_dist;
            std::vector<merge_entry_t> merges;

            queue_entry_t(double costs_, double sum_dist_) :
                costs(costs_),
                sum_dist(sum_dist_) {}
        };

        struct queue_entry_compare {
            bool operator()(const queue_entry_t& a, const queue_entry_t& b) {
                return a.costs > b.costs;
            }
        };

        using queue_t = std::priority_queue<queue_entry_t, std::vector<queue_entry_t>, queue_entry_compare>;

        const std::size_t                _ylength;
        const std::size_t                _depth;
        double                           _max_error;
        const calc_t*                    _base;
        std::shared_ptr<idx_ngram_map_t> _idxmap;
        index_t*                         _index;
        std::mt19937                     _rng;
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
                generate_queue_entries(l + 1, idx << 1, queue);
            }

            // now run merge loop
            while (!queue.empty()) {
                auto best = queue.top();
                queue.pop();

                bool all_notnull = std::all_of(best.merges.begin(), best.merges.end(), [this](const auto& obj){
                    return this->_transformer.levels[obj.l][obj.idx] != nullptr;
                });

                // check if node was already merged and if distance is low enough
                if (all_notnull) {
                    // check if merge would be in error range (i.e. does not increase error beyond _max_error)
                    bool in_error_range = false;
                    if (_transformer.superroot->error + best.sum_dist < _max_error) {
                        in_error_range = true;
                    } else if (error_is_approx) {
                        // ok, error is an approximation anyway, so lets calculate the right value and try again
                        _transformer.superroot->error = calc_error(i);
                        error_is_approx = false;
                        if (_transformer.superroot->error + best.sum_dist < _max_error) {
                            in_error_range = true;
                        }
                    }

                    if (in_error_range) {
                        // adjust error
                        _transformer.superroot->error += best.sum_dist;
                        error_is_approx = true;

                        // execute merges
                        for (const auto& m : best.merges) {
                            _transformer.link_to_parent(m.neighbor, m.l, m.idx);
                            delete _transformer.levels[m.l][m.idx];
                            _transformer.levels[m.l][m.idx] = nullptr;
                        }

                        // generate new queue entries
                        // WARNING: do NOT merge with the merge loop, otherwise it won't generate all queue entries!
                        for (const auto& m : best.merges) {
                            generate_queue_entries(m.l, m.idx, queue);
                        }
                    }
                }
            }
        }

        void generate_queue_entries(std::size_t l_done, std::size_t idx_done, queue_t& queue) {
            std::size_t idx_even = idx_done - (idx_done % 2);
            if ((l_done == _depth)
                || ((l_done > 0) && (_transformer.levels[l_done][idx_even] == nullptr) && (_transformer.levels[l_done][idx_even + 1] == nullptr))) {
                generate_queue_entries_single(l_done - 1, idx_even >> 1, queue);
            }

            std::size_t idx_4 = idx_done - (idx_done % 4);
            if ((l_done == _depth)
                || ((l_done > 1) && (_transformer.levels[l_done][idx_4] == nullptr) && (_transformer.levels[l_done][idx_4 + 1] == nullptr) && (_transformer.levels[l_done][idx_4 + 2] == nullptr) && (_transformer.levels[l_done][idx_4 + 3] == nullptr))) {
                generate_queue_entries_pair(l_done - 1, idx_4 >> 1, queue);
            }

            std::size_t idx_8 = idx_done - (idx_done % 8);
            if ((l_done == _depth) || ((l_done > 2) && (_transformer.levels[l_done][idx_8] == nullptr) && (_transformer.levels[l_done][idx_8 + 1] == nullptr) && (_transformer.levels[l_done][idx_8 + 2] == nullptr) && (_transformer.levels[l_done][idx_8 + 3] == nullptr) && (_transformer.levels[l_done][idx_8 + 4] == nullptr) && (_transformer.levels[l_done][idx_8 + 5] == nullptr) && (_transformer.levels[l_done][idx_8 + 6] == nullptr) && (_transformer.levels[l_done][idx_8 + 7] == nullptr))) {
                generate_queue_entries_4(l_done - 1, idx_8 >> 1, queue);
            }
        }

        void generate_queue_entries_single(std::size_t l, std::size_t idx, queue_t& queue) {
            node_t* current_node = _transformer.levels[l][idx];

            if (current_node != nullptr) {
                auto& current_index_slot = _index->levels[l][idx];
                auto& bucket = current_index_slot.get_bucket(current_node);
                auto neighbors = bucket.get_nearest(current_node, _max_error - _transformer.superroot->error, 1);
                if (!neighbors.empty()) {
                    queue_entry_t qe(neighbors[0].second, neighbors[0].second);
                    qe.merges.emplace_back(l, idx, neighbors[0].first);
                    queue.emplace(std::move(qe));
                }
            }
        }

        void generate_queue_entries_pair(std::size_t l, std::size_t idx, queue_t& queue) {
            node_t* current_node_l  = _transformer.levels[l    ][idx     ];
            node_t* current_node_r  = _transformer.levels[l    ][idx + 1 ];
            node_t* current_node_up = _transformer.levels[l - 1][idx >> 1];

            // upper node will also be not null at thes point
            if (current_node_l != nullptr && current_node_r != nullptr) {
                auto& current_index_slot = _index->levels[l - 1][idx >> 1];
                auto& bucket = current_index_slot.get_bucket2(current_node_up);
                auto neighbors = bucket.get_nearest(
                    current_node_up,
                    _max_error - _transformer.superroot->error,
                    1
                );

                if (!neighbors.empty()) {
                    queue_entry_t qe(neighbors[0].second / 3.0, neighbors[0].second);
                    qe.merges.emplace_back(l    , idx     , neighbors[0].first->child_l);
                    qe.merges.emplace_back(l    , idx + 1 , neighbors[0].first->child_r);
                    qe.merges.emplace_back(l - 1, idx >> 1, neighbors[0].first);
                    queue.emplace(std::move(qe));
                }
            }
        }

        void generate_queue_entries_4(std::size_t l, std::size_t idx, queue_t& queue) {
            node_t* current_node_0  = _transformer.levels[l    ][idx     ];
            node_t* current_node_1  = _transformer.levels[l    ][idx + 1 ];
            node_t* current_node_2  = _transformer.levels[l    ][idx + 2 ];
            node_t* current_node_3  = _transformer.levels[l    ][idx + 3 ];
            node_t* current_node_up = _transformer.levels[l - 2][idx >> 2];

            // upper node will also be not null at thes point
            if (current_node_0 != nullptr && current_node_1 != nullptr && current_node_2 != nullptr && current_node_3 != nullptr) {
                auto& current_index_slot = _index->levels[l - 1][idx >> 1];
                auto& bucket = current_index_slot.get_bucket4(current_node_up);
                auto neighbors = bucket.get_nearest(
                    current_node_up,
                    _max_error - _transformer.superroot->error,
                    1
                );

                if (!neighbors.empty()) {
                    queue_entry_t qe(neighbors[0].second / 7.0, neighbors[0].second);
                    qe.merges.emplace_back(l    , idx     , neighbors[0].first->child_l->child_l);
                    qe.merges.emplace_back(l    , idx + 1 , neighbors[0].first->child_l->child_r);
                    qe.merges.emplace_back(l    , idx + 2 , neighbors[0].first->child_r->child_l);
                    qe.merges.emplace_back(l    , idx + 3 , neighbors[0].first->child_r->child_r);
                    qe.merges.emplace_back(l - 1, idx >> 1, neighbors[0].first->child_l);
                    qe.merges.emplace_back(l - 1, (idx >> 1) + 1, neighbors[0].first->child_r);
                    qe.merges.emplace_back(l - 2, idx >> 2, neighbors[0].first);
                    queue.emplace(std::move(qe));
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
                    node_t* current_node = _transformer.levels[l][idx];

                    // node might already be merged
                    if (current_node != nullptr) {
                        auto& current_index_slot = _index->levels[l][idx];

                        auto& bucket = current_index_slot.get_bucket(current_node);
                        bucket.insert(current_node);

                        if ((current_node->child_l != nullptr) && (current_node->child_r != nullptr)) {
                            auto& bucket2 = current_index_slot.get_bucket2(current_node);
                            bucket2.insert(current_node);
                        }
                        if (current_node->child_l != nullptr && current_node->child_r != nullptr && current_node->child_l->child_l != nullptr && current_node->child_l->child_r != nullptr && current_node->child_r->child_l != nullptr && current_node->child_r->child_r != nullptr) {
                            auto& bucket4 = current_index_slot.get_bucket4(current_node);
                            bucket4.insert(current_node);
                        }

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

double calc_compression_rate(const index_t* index, year_t ylength, std::size_t n) {
    std::size_t size_normal = sizeof(calc_t) * static_cast<std::size_t>(ylength) * n;

    // don't count the size for storing the actual ngram because that's additional information
    // and not strictly part of the compression
    std::size_t size_compression = sizeof(node_t) * index->node_count
        + (sizeof(superroot_t) - sizeof(ngram_t)) * n;

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
    index_t index(depth, n);
    engine eng(ylength, depth, max_error, base, idxmap, &index);

    std::cout << "build and merge trees" << std::endl;
    std::mt19937 rng;
    std::size_t counter = 0;
    std::vector<std::size_t> indices(n);
    std::generate(indices.begin(), indices.end(), [&counter](){
        return counter++;
    });
    std::shuffle(indices.begin(), indices.end(), rng);

    std::size_t n_test = 100000; // DEBUG
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
