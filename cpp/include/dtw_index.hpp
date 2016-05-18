#pragma once

#include <cstdint>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/adapted/boost_array.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/cartesian/distance_pythagoras.hpp>
#include <boost/interprocess/allocators/adaptive_pool.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>

BOOST_GEOMETRY_REGISTER_BOOST_ARRAY_CS(cs::cartesian)

#include "dtw.hpp"

constexpr std::size_t dtw_index_resolution       = 64;
constexpr std::size_t dtw_index_resolution_shift = 6;
static_assert(is_power_of_2(dtw_index_resolution), "DTW resolution has to be a power of 2!");
static_assert(dtw_index_resolution >= 2, "DTW resolution has to be at least 2!");
static_assert(1u << dtw_index_resolution_shift == dtw_index_resolution, "DTW resolution and the bitshift variable are not consistent!");

constexpr std::size_t tree_max_node_size         = 1024 * 64;
constexpr std::size_t alloc_pool_size            = 1024 * 512;

using segment_manager_t  = boost::interprocess::managed_mapped_file::segment_manager;

using point_t            = boost::array<calc_t, dtw_index_resolution>;
using box_t              = boost::geometry::model::box<point_t>;
using node_t             = std::pair<box_t, std::size_t>;
using indexable_t        = boost::geometry::index::indexable<node_t>;

struct equal_to_t {
    bool operator()(const node_t& a, const node_t& b) const {
        return a.second == b.second;
    }
};

using params_t           = boost::geometry::index::quadratic<tree_max_node_size>;
using allocator_node_t   = boost::interprocess::adaptive_pool<node_t, segment_manager_t, alloc_pool_size>;
using tree_t             = boost::geometry::index::rtree<node_t, params_t, indexable_t, equal_to_t, allocator_node_t>;

using allocator_point_t  = boost::interprocess::adaptive_pool<point_t, segment_manager_t, alloc_pool_size>;
using downstorage_t      = boost::interprocess::vector<point_t, allocator_point_t>;


calc_t mindist_unnorm(const box_t& a, const box_t& b) {
    return boost::geometry::comparable_distance(a, b);
}

calc_t lb_paa_unnorm(const box_t& a, const point_t& b) {
    return boost::geometry::comparable_distance(a, b);
}

template <typename T>
struct get_downsampled_generic {
    static point_t f(const calc_t* local_base, year_t ylength, point_t& result) {
        std::size_t rate = ylength >> dtw_index_resolution_shift;

        for (std::size_t idx = 0, input_idx = 0; idx < dtw_index_resolution; ++idx, input_idx += rate) {
            result[idx] = std::accumulate(
                &local_base[input_idx],
                &local_base[input_idx + rate],
                T::init(),
                T::reduce
            );
        }

        if (T::needs_normalization()) {
            calc_t norm_factor = T::calc_normfactor(ylength);
            for (auto& x : result) {
                x *= norm_factor;
            }
        }

        return result;
    }
};

struct get_downsampled_data_impl {
    static calc_t init() {
        return 0.0;
    }

    static calc_t reduce(calc_t a, calc_t b) {
        return a + b;
    }

    static bool needs_normalization() {
        return true;
    }

    static calc_t calc_normfactor(year_t ylength) {
        return static_cast<calc_t>(dtw_index_resolution) / static_cast<calc_t>(ylength);
    }
};

struct get_downsampled_l_impl {
    static calc_t init() {
        return std::numeric_limits<calc_t>::infinity();
    }

    static calc_t reduce(calc_t a, calc_t b) {
        return std::min(a, b);
    }

    static bool needs_normalization() {
        return false;
    }

    static calc_t calc_normfactor(year_t) {
        return 0.0;
    }
};

struct get_downsampled_u_impl {
    static calc_t init() {
        return -std::numeric_limits<calc_t>::infinity();
    }

    static calc_t reduce(calc_t a, calc_t b) {
        return std::max(a, b);
    }

    static bool needs_normalization() {
        return false;
    }

    static calc_t calc_normfactor(year_t) {
        return 0.0;
    }
};

using get_downsampled_data = get_downsampled_generic<get_downsampled_data_impl>;
using get_downsampled_l    = get_downsampled_generic<get_downsampled_l_impl>;
using get_downsampled_u    = get_downsampled_generic<get_downsampled_u_impl>;


std::pair<std::vector<calc_t>, std::vector<calc_t>> get_lu(const calc_t* base, year_t ylength, std::size_t i, std::size_t r) {
    std::vector<calc_t> l(ylength, 0.0);
    std::vector<calc_t> u(ylength, 0.0);
    const calc_t* local_base = base + (i * ylength);

    for (std::size_t idx = 0; idx < ylength; ++idx) {
        std::size_t idx2_min = 0;
        std::size_t idx2_max = ylength - 1;
        if (idx > r) {
            idx2_min = idx - r;
        }
        if (idx < ylength - r - 1) {
            idx2_max = idx + r;
        }

        calc_t element_l = std::numeric_limits<calc_t>::infinity();
        calc_t element_u = -std::numeric_limits<calc_t>::infinity();
        for (std::size_t idx2 = idx2_min; idx2 <= idx2_max; ++idx2) {
            element_l = std::min(element_l, local_base[idx2]);
            element_u = std::max(element_u, local_base[idx2]);
        }

        l[idx] = element_l;
        u[idx] = element_u;
    }

    return std::make_pair(std::move(l), std::move(u));
}
