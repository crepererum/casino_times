#pragma once

#include <memory>

#include <boost/functional/hash.hpp>
#include <boost/interprocess/allocators/adaptive_pool.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/allocators/node_allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/locale.hpp>

#include "parser.hpp"

constexpr std::size_t alloc_nodes = 1024 * 1024;

using mapped_file_ptr_t         = std::shared_ptr<boost::interprocess::managed_mapped_file>;

struct node_t;
struct superroot_t;

using node_ptr_t                = boost::interprocess::offset_ptr<node_t>;
using superroot_ptr_t           = boost::interprocess::offset_ptr<superroot_t>;

using segment_manager_t         = boost::interprocess::managed_mapped_file::segment_manager;

template <typename T>
using allocator_t               = boost::interprocess::node_allocator<T, segment_manager_t, alloc_nodes>;

template <typename T>
using allocator_adaptive_t      = boost::interprocess::adaptive_pool<T, segment_manager_t, alloc_nodes>;

using allocator_node_t          = allocator_t<node_t>;
using allocator_superroot_t     = allocator_t<superroot_t>;
using allocator_superroot_ptr_t = allocator_adaptive_t<superroot_ptr_t>;

using superroot_vector_t        = boost::interprocess::vector<superroot_ptr_t, allocator_superroot_ptr_t>;

using inexact_t = float;

struct superroot_t {
    node_ptr_t root;
    inexact_t  approx;
    inexact_t  error;
};

struct __attribute__((packed)) node_t {
    node_ptr_t child_l;
    node_ptr_t child_r;
    inexact_t  x;
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
    struct hash<std::pair<node_ptr_t, node_ptr_t>> {
        size_t operator()(const std::pair<node_ptr_t, node_ptr_t>& obj) const {
            std::size_t seed = 0;
            boost::hash_combine(seed, obj.first.get());
            boost::hash_combine(seed, obj.second.get());
            return seed;
        }
    };
}

template <typename T>
boost::interprocess::offset_ptr<T> alloc_in_mapped_file(allocator_t<T>& alloc) {
    return alloc.allocate(1);
}

template <typename T>
void dealloc_in_mapped_file(allocator_t<T>& alloc, const boost::interprocess::offset_ptr<T>& ptr) {
    alloc.deallocate(ptr.get(), 1);
}
