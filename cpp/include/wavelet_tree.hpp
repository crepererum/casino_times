#pragma once

#include <memory>

#include <boost/functional/hash.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/locale.hpp>

#include "parser.hpp"

using mapped_file_ptr_t     = std::shared_ptr<boost::interprocess::managed_mapped_file>;

struct node_t;
struct superroot_t;

using node_ptr_t            = boost::interprocess::offset_ptr<node_t>;
using superroot_ptr_t       = boost::interprocess::offset_ptr<superroot_t>;

using segment_manager_t     = boost::interprocess::managed_mapped_file::segment_manager;
using allocator_superroot_t = boost::interprocess::allocator<superroot_ptr_t, segment_manager_t>;
using superroot_vector_t    = boost::interprocess::vector<superroot_ptr_t, allocator_superroot_t>;

struct superroot_t {
    node_ptr_t root;
    calc_t     approx;
    calc_t     error;
};

struct node_t {
    node_ptr_t child_l;
    node_ptr_t child_r;
    calc_t     x;
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
boost::interprocess::offset_ptr<T> alloc_in_mapped_file(mapped_file_ptr_t& f) {
    return static_cast<T*>(f->allocate(sizeof(T)));
}

template <typename T>
void dealloc_in_mapped_file(mapped_file_ptr_t& f, const boost::interprocess::offset_ptr<T>& ptr) {
    f->deallocate(ptr.get());
}
