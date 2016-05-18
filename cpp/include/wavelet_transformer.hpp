#pragma once

#include "wavelet.hpp"
#include "wavelet_tree.hpp"

class transformer {
    public:
        superroot_ptr_t                      superroot;
        std::vector<std::vector<node_ptr_t>> levels;

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

        superroot_ptr_t data_to_tree(const calc_t* data) {
            _mywt.run_dwt(data);

            superroot = new superroot_t;
            superroot->approx = static_cast<inexact_t>(_mywt.output()[0]);
            superroot->error  = 0;

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t outdelta  = 1 << l;
                std::size_t width     = outdelta;  // same calculation

                levels[l].clear();

                for (std::size_t idx = 0; idx < width; ++idx) {
                    node_ptr_t node  = new node_t;
                    for (std::size_t c = 0; c < n_children; ++c) {
                        node->children[c] = nullptr;
                    }
                    node->x       = static_cast<inexact_t>(_mywt.output()[outdelta + idx] * _influence_sqrt[l]);

                    link_to_parent(node, l, idx);

                    levels[l].push_back(node);
                }
            }

            return superroot;
        }

        void tree_to_data(calc_t* data) {
            // prepare idwt
            _mywt.output()[0] = static_cast<double>(superroot->approx);
            std::vector<node_ptr_t> layer_a{superroot->root};
            std::vector<node_ptr_t> layer_b;
            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t width    = static_cast<std::size_t>(1) << l;
                std::size_t outdelta = width; // same calculation

                layer_b.clear();
                for (std::size_t idx = 0; idx < width; ++idx) {
                    node_ptr_t current_node = layer_a[idx];
                    if (current_node) {
                        _mywt.output()[outdelta + idx] = static_cast<double>(layer_a[idx]->x) / _influence_sqrt[l];
                        layer_b.push_back(layer_a[idx]->children[0]);
                        layer_b.push_back(layer_a[idx]->children[1]);
                    } else {
                        _mywt.output()[outdelta + idx] = 0.0;
                        layer_b.push_back(nullptr);
                        layer_b.push_back(nullptr);
                    }
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
                    parent->children[1] = node;
                } else {
                    parent->children[0] = node;
                }
            }
        }

    private:
        const std::size_t     _ylength;
        const std::size_t     _depth;
        wavelet_transform     _mywt;
        std::vector<double>   _influence_sqrt;
};
