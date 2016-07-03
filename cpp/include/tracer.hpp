#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "wavelet_tree.hpp"

struct tracer_types final {
    struct node_plus_t final {
        node_ptr_t node;
        std::size_t l;
        std::size_t j;
    };

    struct hash_node_plus_t final {
        std::size_t operator()(const node_plus_t& n) const {
            return _helper(n.node);
        }

        std::hash<node_ptr_t> _helper;
    };

    struct equal_to_node_plus_t final {
        bool operator()(const node_plus_t& a, const node_plus_t& b) const {
            return a.node == b.node;
        }
    };

    using next_up_t = std::unordered_map<node_plus_t, float, hash_node_plus_t, equal_to_node_plus_t>;
    using next_down_t = std::vector<node_plus_t>;
};

struct tracer_impl_noop {
    bool chain_up_pre(const tracer_types::next_up_t&) { return true; }
    void chain_up_post() {}
    void chain_down_pre(const tracer_types::next_down_t&) {}
    void chain_down_post() {}

    void up_pre(const tracer_types::next_up_t&) {}
    void up_post(const tracer_types::next_up_t&) {}
    void down_pre(const tracer_types::next_down_t&) {}
    void down_post(const tracer_types::next_down_t&) {}

    bool filter_node_up(const tracer_types::node_plus_t&, float) { return true; }
    bool filter_node_down(const tracer_types::node_plus_t&) { return true; }

    void found_superroot(const superroot_ptr_t&, float) {}
};

template <typename T = tracer_impl_noop>
class tracer final {
    public:
        explicit tracer(const index_stored_t* index, T t = T()) : _index(index), _t(t) {}

        void operator()(std::size_t i) {
            (*this)((*_index->superroots)[i]);
        }

        void operator()(superroot_ptr_t start) {
            _next_down_a.clear();
            _next_down_a.emplace_back(tracer_types::node_plus_t{start->root, 0, 0});

            _t.chain_down_pre(_next_down_a);
            down();
            _t.chain_down_post();
        }

        T& t() {
            return _t;
        }

    private:
        const index_stored_t*     _index;
        T                         _t;
        tracer_types::next_up_t   _next_up_a;
        tracer_types::next_up_t   _next_up_b;
        tracer_types::next_down_t _next_down_a;
        tracer_types::next_down_t _next_down_b;

        void up() {
            _t.up_pre(_next_up_a);

            _next_up_b.clear();
            for (const auto& n : _next_up_a) {
                if (n.first.l > 0) {
                    auto parents = _index->find_parents(n.first.node);
                    if (parents) {
                        float weight = n.second / static_cast<float>(parents->size());
                        for (const auto& p : *parents) {
                            tracer_types::node_plus_t candidate{p, n.first.l - 1, n.first.j / n_children};
                            add_to_next_up(_next_up_b, candidate, weight);
                        }
                    }
                } else {
                    auto superroots = _index->find_superroots(n.first.node);
                    if (superroots) {
                        for (const auto& s : *superroots) {
                            _t.found_superroot(s, n.second);
                        }
                    }
                }
            }

            _t.up_post(_next_up_b);

            std::swap(_next_up_a, _next_up_b);
            if (!_next_up_a.empty()) {
                up();
            }
        }

        void down() {
            _t.down_pre(_next_down_a);

            _next_down_b.clear();
            _next_up_a.clear();
            for (const auto& n : _next_down_a) {
                float weight = 1;// / static_cast<float>(_next_down_a.size());
                add_to_next_up(_next_up_a, n, weight);

                for (std::size_t d = 0; d < n_children; ++d) {
                    const auto& c = n.node->children[d];
                    if (c != nullptr) {
                        tracer_types::node_plus_t candidate{c, n.l + 1, (n.j * n_children) + d};
                        if (_t.filter_node_down(candidate)){
                            _next_down_b.emplace_back(candidate);
                        }
                    }
                }
            }

            if (!_next_up_a.empty() && _t.chain_up_pre(_next_up_a)){
                up();
                _t.chain_up_post();
            }

            _t.down_post(_next_down_b);

            std::swap(_next_down_a, _next_down_b);
            if (!_next_down_a.empty()) {
                down();
            }
        }

        void add_to_next_up(tracer_types::next_up_t& next_up, tracer_types::node_plus_t n, float weight) {
            if (_t.filter_node_up(n, weight)) {
                auto it = next_up.find(n);
                if (it == next_up.end()) {
                    next_up.emplace(std::make_pair(n, weight));
                } else {
                    it->second *= weight;
                }
            }
        }
};

struct tracer_impl_limiter : tracer_impl_noop {
    std::size_t depth;
    std::size_t maxdepth;
    std::size_t begin;
    std::size_t end;
    float       minweight;

    tracer_impl_limiter(
            std::size_t depth_,
            std::size_t maxdepth_,
            std::size_t begin_,
            std::size_t end_,
            float minweight_)
        : depth(depth_),
        maxdepth(maxdepth_),
        begin(begin_),
        end(end_),
        minweight(minweight_) {}

    bool filter_node_up(const tracer_types::node_plus_t&, float weight) {
        return minweight <= weight;
    }

    bool filter_node_down(const tracer_types::node_plus_t& n) {
        std::size_t width = std::size_t(1) << (depth - n.l);
        std::size_t influence_begin = width * n.j;
        std::size_t influence_end = width * (n.j + 1);
        return (n.l < maxdepth) && (end > influence_begin) && (begin < influence_end);
    }
};

