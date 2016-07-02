#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/locale.hpp>

#include <half.hpp>

#include "short_offset_ptr.hpp"
#include "parser.hpp"

using half_float::half;

constexpr std::size_t alloc_nodes = 1024 * 1024;
constexpr std::size_t n_children  = 2;

using mapped_file_ptr_t         = std::shared_ptr<boost::interprocess::managed_mapped_file>;

struct node_t;
struct superroot_t;

using node_ptr_t                = short_offset_ptr<node_t>;
using superroot_ptr_t           = short_offset_ptr<superroot_t>;

using segment_manager_t         = boost::interprocess::managed_mapped_file::segment_manager;

template <typename T>
using allocator_t               = boost::interprocess::allocator<T, segment_manager_t>;

template <typename T>
using allocator_adaptive_t      = boost::interprocess::allocator<T, segment_manager_t>;

using allocator_node_t          = allocator_t<node_t>;
using allocator_superroot_t     = allocator_t<superroot_t>;
using allocator_node_ptr_t      = allocator_adaptive_t<node_ptr_t>;
using allocator_superroot_ptr_t = allocator_adaptive_t<superroot_ptr_t>;

using node_vector_t             = std::vector<node_ptr_t, allocator_node_ptr_t>;
using superroot_vector_t        = std::vector<superroot_ptr_t, allocator_superroot_ptr_t>;

using allocator_node_vector_t   = allocator_adaptive_t<node_vector_t>;
using allocator_superroot_vector_t = allocator_adaptive_t<superroot_vector_t>;

using parents_table_t           = std::unordered_map<node_ptr_t, node_vector_t, offset_hash<node_t>, std::equal_to<node_ptr_t>, allocator_node_vector_t>;
using superroots_table_t        = std::unordered_map<node_ptr_t, superroot_vector_t, offset_hash<node_t>, std::equal_to<node_ptr_t>, allocator_superroot_vector_t>;

using children_t                = std::array<node_ptr_t, n_children>;

using inexact_t = float;
using approx_t  = half;

struct superroot_t {
    node_ptr_t root;
    approx_t   approx;
    inexact_t  error;
};

struct __attribute__((packed)) node_t {
    children_t children;
    approx_t   x;
};

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
    struct hash<superroot_ptr_t> {
        size_t operator()(const superroot_ptr_t& obj) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, obj.get());
            return seed;
        }
    };

    template<>
    struct hash<children_t> {
        size_t operator()(const children_t& obj) const {
            std::size_t seed = 0;
            for (std::size_t c = 0; c < n_children; ++c) {
                boost::hash_combine(seed, obj[c].get());
            }
            return seed;
        }
    };
}

template <typename T>
short_offset_ptr<T> alloc_in_mapped_file(allocator_t<T>& alloc) {
    return alloc.allocate(1).get();
}

template <typename T>
void dealloc_in_mapped_file(allocator_t<T>& alloc, const short_offset_ptr<T>& ptr) {
    alloc.deallocate(ptr.get(), 1);
}

struct index_stored_t {
    superroot_vector_t* superroots;
    parents_table_t*    parents_table;
    superroots_table_t* superroots_table;

    allocator_node_ptr_t         alloc_node_ptr;
    allocator_superroot_ptr_t    alloc_superroot_ptr;
    allocator_node_vector_t      alloc_node_vector;
    allocator_superroot_vector_t alloc_superroot_vector;

    index_stored_t(std::shared_ptr<boost::interprocess::managed_mapped_file> findex, std::size_t n)
        : alloc_node_ptr(findex->get_segment_manager()),
        alloc_superroot_ptr(findex->get_segment_manager()),
        alloc_node_vector(findex->get_segment_manager()),
        alloc_superroot_vector(findex->get_segment_manager()) {

        short_offset_ptr<std::uint8_t> anchor{findex->find_or_construct<std::uint8_t>("hash_anchor")()};

        superroots    = findex->find_or_construct<superroot_vector_t>("superroots")(
            n,
            alloc_superroot_ptr
        );

        parents_table = findex->find_or_construct<parents_table_t>("parents_table")(
            0,
            offset_hash<node_t>{anchor},
            std::equal_to<node_ptr_t>{},
            alloc_node_vector
        );

        superroots_table = findex->find_or_construct<superroots_table_t>("superroots_table")(
            0,
            offset_hash<node_t>{anchor},
            std::equal_to<node_ptr_t>{},
            alloc_superroot_vector
        );
    }

    void register_parent(const node_ptr_t& node, const node_ptr_t& parent) {
        auto it = parents_table->find(node);
        if (it == parents_table->end()) {
            bool tmp;
            std::tie(it, tmp) = parents_table->emplace(std::make_pair(node, node_vector_t{0, alloc_node_ptr}));
        }
        it->second.emplace_back(parent);
    }

    void register_superroot(const node_ptr_t& node, const superroot_ptr_t& superroot) {
        auto it = superroots_table->find(node);
        if (it == superroots_table->end()) {
            bool tmp;
            std::tie(it, tmp) = superroots_table->emplace(std::make_pair(node, superroot_vector_t{0, alloc_superroot_ptr}));
        }
        it->second.emplace_back(superroot);
    }

    const node_vector_t* find_parents(const node_ptr_t& node) const {
        auto it = parents_table->find(node);
        if (it != parents_table->end()) {
            return &(it->second);
        } else {
            return nullptr;
        }
    }

    const superroot_vector_t* find_superroots(const node_ptr_t& node) const {
        auto it = superroots_table->find(node);
        if (it != superroots_table->end()) {
            return &(it->second);
        } else {
            return nullptr;
        }
    }

    void delete_all_ptrs(const std::shared_ptr<boost::interprocess::managed_mapped_file>& findex) {
        findex->destroy_ptr(superroots);
        findex->destroy_ptr(parents_table);
        findex->destroy_ptr(superroots_table);
    }
};
