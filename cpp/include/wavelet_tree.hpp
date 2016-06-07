#pragma once

#include <array>
#include <memory>

#include <boost/functional/hash.hpp>
#include <boost/interprocess/allocators/adaptive_pool.hpp>
#include <boost/interprocess/allocators/private_node_allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/locale.hpp>
#include <boost/unordered_map.hpp>

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
using allocator_t               = boost::interprocess::private_node_allocator<T, segment_manager_t, alloc_nodes>;

template <typename T>
using allocator_adaptive_t      = boost::interprocess::adaptive_pool<T, segment_manager_t, alloc_nodes>;

using allocator_node_t          = allocator_t<node_t>;
using allocator_superroot_t     = allocator_t<superroot_t>;
using allocator_node_ptr_t      = allocator_adaptive_t<node_ptr_t>;
using allocator_superroot_ptr_t = allocator_adaptive_t<superroot_ptr_t>;

using node_vector_t             = boost::interprocess::vector<node_ptr_t, allocator_node_ptr_t>;
using superroot_vector_t        = boost::interprocess::vector<superroot_ptr_t, allocator_superroot_ptr_t>;

using allocator_node_vector_t   = allocator_adaptive_t<node_vector_t>;
using allocator_superroot_vector_t = allocator_adaptive_t<superroot_vector_t>;

using parents_table_t           = boost::unordered_map<node_ptr_t, node_vector_t, offset_hash<node_t>, std::equal_to<node_ptr_t>, allocator_node_vector_t>;
using superroots_table_t        = boost::unordered_map<node_ptr_t, superroot_vector_t, offset_hash<node_t>, std::equal_to<node_ptr_t>, allocator_superroot_vector_t>;

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
