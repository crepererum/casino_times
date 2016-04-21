#pragma once

#include "wavelet.hpp"
#include "wavelet_tree.hpp"

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
            _mapped_file(mapped_file),
            _alloc_node(_mapped_file->get_segment_manager()),
            _alloc_superroot(_mapped_file->get_segment_manager())
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

            superroot = alloc_in_mapped_file<superroot_t>(_alloc_superroot);
            superroot->approx = _mywt.output()[0];
            superroot->error  = 0;

            for (std::size_t l = 0; l < _depth; ++l) {
                std::size_t outdelta  = 1 << l;
                std::size_t width     = outdelta;  // same calculation

                levels[l].clear();

                for (std::size_t idx = 0; idx < width; ++idx) {
                    node_ptr_t node  = alloc_in_mapped_file<node_t>(_alloc_node);
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
        const std::size_t     _ylength;
        const std::size_t     _depth;
        wavelet_transform     _mywt;
        std::vector<double>   _influence_sqrt;
        mapped_file_ptr_t     _mapped_file;
        allocator_node_t      _alloc_node;
        allocator_superroot_t _alloc_superroot;
};
