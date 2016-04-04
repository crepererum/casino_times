#pragma once

#include <cstdint>

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

#include "parser.hpp"
#include "utils.hpp"

#define SIMDPP_ARCH_X86_AVX2
#define SIMDPP_NO_DISPATCHER
#include <simdpp/simd.h>
#pragma message "optimal vector size: " XSTR(SIMDPP_FAST_FLOAT64_SIZE)

using calc_t = double;
static_assert(sizeof(double) == 8, "double isn't 64bit :(");

calc_t dist(calc_t a, calc_t b) {
    calc_t d = a - b;
    return d * d;
}

template <unsigned N>
simdpp::float64<N> dist(simdpp::float64<N> a, simdpp::float64<N> b) {
    simdpp::float64<N> d = sub(a, b);
    return mul(d, d);
}

template <typename T>
class dtw_generic {
    public:
        static constexpr std::size_t n = T::n;

        using internal_t      = typename T::internal_t;
        using indices_t       = typename T::indices_t;

        dtw_generic(const calc_t* base, year_t ylength, std::size_t r) :
            _r(r),
            _r2(_r << 1),
            _r2_plus(_r2 + 1),
            _base(base),
            _ylength(ylength),
            _ylength_minus(_ylength - 1),
            _ylength_minus_r(_ylength - _r),
            _store_a(_r2_plus),
            _store_b(_r2_plus),
            _store_tmp(ylength) {}

        internal_t calc(std::size_t i, indices_t j0) {
            const calc_t* local_base_i = _base + (i * _ylength);
            load_to_tmp(j0);

            std::fill(_store_a.begin(), _store_a.end(), T::infinity());
            T::store(&_store_a[_r], T::zero());

            // remember: r <= ylength/2

            for (std::size_t idx_i = 0; idx_i < _r; ++idx_i) {
                std::size_t idx_j_min = 0;
                std::size_t idx_j_max = idx_i + _r;

                inner_loop(local_base_i, idx_i, idx_j_min, idx_j_max, _r - idx_i);
            }

            for (std::size_t idx_i = _r; idx_i < _ylength_minus_r; ++idx_i) {
                std::size_t idx_j_min = idx_i - _r;
                std::size_t idx_j_max = idx_i + _r;

                inner_loop(local_base_i, idx_i, idx_j_min, idx_j_max, 0);
            }

            for (std::size_t idx_i = _ylength_minus_r; idx_i < _ylength; ++idx_i) {
                std::size_t idx_j_min = idx_i - _r;
                std::size_t idx_j_max = _ylength_minus;

                inner_loop(local_base_i, idx_i, idx_j_min, idx_j_max, 0);
            }

            return T::load(&_store_a[_r]);
        }

    private:
        using bases_t         = typename T::bases_t;
        using store_t         = std::vector<internal_t, boost::alignment::aligned_allocator<internal_t, T::alignment>>;
        using difference_type = typename store_t::difference_type;

        const std::size_t   _r;
        const std::size_t   _r2;
        const std::size_t   _r2_plus;
        const calc_t* const _base;
        const year_t        _ylength;
        const year_t        _ylength_minus;
        const std::size_t   _ylength_minus_r;
        store_t             _store_a;
        store_t             _store_b;
        store_t             _store_tmp;

        void inner_loop(const calc_t* local_base_i, std::size_t idx_i, std::size_t idx_j_min, std::size_t idx_j_max, std::size_t store_delta) {
            std::size_t store_pos = store_delta;

            internal_t current_value   = T::infinity();
            internal_t last_value_here = T::infinity();  // will never be used
            internal_t last_value_next = T::load(&_store_a[store_pos]);

            std::fill(
                _store_b.begin(),
                _store_b.begin() + static_cast<difference_type>(store_pos),
                T::infinity()
            );

            for (std::size_t idx_j = idx_j_min; idx_j <= idx_j_max; ++idx_j) {
                internal_t a = T::convert_single(local_base_i[idx_i]);
                internal_t b = T::load(&_store_tmp[idx_j]);
                internal_t cost = dist(a, b);

                last_value_here = last_value_next;
                if (store_pos < _r2) {
                    last_value_next = T::load(&_store_a[store_pos + 1]);
                } else {
                    last_value_next = T::infinity();
                }

                // rename variables (let's hope the compiler optimizes that)
                internal_t dtw_insert = last_value_next;
                internal_t dtw_delete = current_value;  // value from last round
                internal_t dtw_match  = last_value_here;

                current_value = cost + T::min3(dtw_insert, dtw_delete, dtw_match);
                T::store(&_store_b[store_pos], current_value);

                ++store_pos;
            }

            std::fill(
                _store_b.begin() + static_cast<difference_type>(store_pos),
                _store_b.end(),
                T::infinity()
            );

            std::swap(_store_a, _store_b);
        }

        void load_to_tmp(indices_t j0) {
            bases_t local_bases_j = T::get_bases(_base, _ylength, j0);
            for (std::size_t idx = 0; idx < _ylength; ++idx) {
                T::store(&_store_tmp[idx], T::convert_multiple(local_bases_j, idx));
            }
        }
};

struct dtw_impl_simple {
    static constexpr std::size_t n         = 1;
    static constexpr std::size_t alignment = n * 8;

    using internal_t = calc_t;
    using indices_t  = std::size_t;
    using bases_t    = const calc_t*;

    inline static internal_t convert_single(calc_t x) {
        return static_cast<calc_t>(x);
    }

    inline static internal_t convert_multiple(bases_t bases, std::size_t idx) {
        return static_cast<calc_t>(bases[idx]);
    }

    inline static bases_t get_bases(const calc_t* base, std::size_t ylength, indices_t j0) {
        return base + (j0 * ylength);
    }

    inline static internal_t load(internal_t* ptr) {
        return *ptr;
    }

    inline static internal_t min3(internal_t a, internal_t b, internal_t c) {
        return std::min(a, std::min(b, c));
    }

    inline static internal_t infinity() {
        return std::numeric_limits<calc_t>::infinity();
    }

    inline static void store(internal_t* ptr, internal_t data) {
        *ptr = data;
    }

    inline static internal_t zero() {
        return 0.0;
    }
};

struct dtw_impl_vectorized {
    // Officially SIMDPP_FAST_FLOAT64_SIZE should be optimal,
    // but higher counts seem to boost performance even higher.
    // This value is bisected, but might need to be re-adjusted for
    // other architectures or when multi-threading is implemented.
    static constexpr std::size_t n         = 16;
    static constexpr std::size_t alignment = n * 8;
    static_assert(n >= SIMDPP_FAST_FLOAT64_SIZE, "dtw_impl_vectorized is not optimal!");

    using internal_t = simdpp::float64<n>;
    using bases_t    = std::array<const calc_t*, n>;

    inline static internal_t convert_single(calc_t x) {
        return simdpp::make_float(x);
    }

    inline static internal_t convert_multiple(bases_t bases, std::size_t idx) {
        return simdpp::make_float(
            std::get< 0>(bases)[idx],
            std::get< 1>(bases)[idx],
            std::get< 2>(bases)[idx],
            std::get< 3>(bases)[idx],
            std::get< 4>(bases)[idx],
            std::get< 5>(bases)[idx],
            std::get< 6>(bases)[idx],
            std::get< 7>(bases)[idx],
            std::get< 8>(bases)[idx],
            std::get< 9>(bases)[idx],
            std::get<10>(bases)[idx],
            std::get<11>(bases)[idx],
            std::get<12>(bases)[idx],
            std::get<13>(bases)[idx],
            std::get<14>(bases)[idx],
            std::get<15>(bases)[idx]
        );
    }

    inline static internal_t load(internal_t* ptr) {
        return simdpp::load(ptr);
    }

    inline static internal_t min3(internal_t a, internal_t b, internal_t c) {
        return simdpp::min(a, simdpp::min(b, c));
    }

    inline static internal_t infinity() {
        return simdpp::make_float(std::numeric_limits<double>::infinity());
    }

    inline static void store(internal_t* ptr, internal_t data) {
        simdpp::store(ptr, data);
    }

    inline static internal_t zero() {
        return simdpp::make_float(0.0);
    }
};

struct dtw_impl_vectorized_linear : dtw_impl_vectorized {
    using indices_t = std::size_t;

    inline static bases_t get_bases(const calc_t* base, std::size_t ylength, indices_t j0) {
        return {{
            (base + ((j0 +  0) * ylength)),
            (base + ((j0 +  1) * ylength)),
            (base + ((j0 +  2) * ylength)),
            (base + ((j0 +  3) * ylength)),
            (base + ((j0 +  4) * ylength)),
            (base + ((j0 +  5) * ylength)),
            (base + ((j0 +  6) * ylength)),
            (base + ((j0 +  7) * ylength)),
            (base + ((j0 +  8) * ylength)),
            (base + ((j0 +  9) * ylength)),
            (base + ((j0 + 10) * ylength)),
            (base + ((j0 + 11) * ylength)),
            (base + ((j0 + 12) * ylength)),
            (base + ((j0 + 13) * ylength)),
            (base + ((j0 + 14) * ylength)),
            (base + ((j0 + 15) * ylength))
        }};
    }
};

struct dtw_impl_vectorized_shuffled : dtw_impl_vectorized {
    using indices_t = const std::array<std::size_t, n>&;

    inline static bases_t get_bases(const calc_t* base, std::size_t ylength, indices_t j0) {
        return {{
            (base + (std::get< 0>(j0) * ylength)),
            (base + (std::get< 1>(j0) * ylength)),
            (base + (std::get< 2>(j0) * ylength)),
            (base + (std::get< 3>(j0) * ylength)),
            (base + (std::get< 4>(j0) * ylength)),
            (base + (std::get< 5>(j0) * ylength)),
            (base + (std::get< 6>(j0) * ylength)),
            (base + (std::get< 7>(j0) * ylength)),
            (base + (std::get< 8>(j0) * ylength)),
            (base + (std::get< 9>(j0) * ylength)),
            (base + (std::get<10>(j0) * ylength)),
            (base + (std::get<11>(j0) * ylength)),
            (base + (std::get<12>(j0) * ylength)),
            (base + (std::get<13>(j0) * ylength)),
            (base + (std::get<14>(j0) * ylength)),
            (base + (std::get<15>(j0) * ylength)),
        }};
    }
};

using dtw_simple              = dtw_generic<dtw_impl_simple>;
using dtw_vectorized_linear   = dtw_generic<dtw_impl_vectorized_linear>;
using dtw_vectorized_shuffled = dtw_generic<dtw_impl_vectorized_shuffled>;
