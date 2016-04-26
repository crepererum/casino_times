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
#define SIMDPP_ARCH_X86_FMA3
#define SIMDPP_NO_DISPATCHER
#include <simdpp/simd.h>
#pragma message "optimal vector size: " XSTR(SIMDPP_FAST_FLOAT64_SIZE)

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
        using source_t        = typename T::source_t;

        dtw_generic(source_t source, std::size_t length, std::size_t r, T t = T()) :
            _r(r),
            _r2(_r << 1),
            _r2_plus(_r2 + 1),
            _source(source),
            _length(length),
            _length_minus(_length - 1),
            _length_minus_r(_length - _r),
            _store_a(_r2_plus),
            _store_b(_r2_plus),
            _store_tmp(_length),
            _t(t) {}

        internal_t calc(std::size_t i, indices_t j0) {
            base_t local_base_i = _t.get_base(_source, _length, i);
            load_to_tmp(j0);

            std::fill(_store_a.begin(), _store_a.end(), _t.infinity());
            _t.store(&_store_a[_r], _t.zero());

            // remember: r <= length/2

            for (std::size_t idx_i = 0; idx_i < _r; ++idx_i) {
                std::size_t idx_j_min = 0;
                std::size_t idx_j_max = idx_i + _r;

                inner_loop(local_base_i, idx_i, idx_j_min, idx_j_max, _r - idx_i);
            }

            for (std::size_t idx_i = _r; idx_i < _length_minus_r; ++idx_i) {
                std::size_t idx_j_min = idx_i - _r;
                std::size_t idx_j_max = idx_i + _r;

                inner_loop(local_base_i, idx_i, idx_j_min, idx_j_max, 0);
            }

            for (std::size_t idx_i = _length_minus_r; idx_i < _length; ++idx_i) {
                std::size_t idx_j_min = idx_i - _r;
                std::size_t idx_j_max = _length_minus;

                inner_loop(local_base_i, idx_i, idx_j_min, idx_j_max, 0);
            }

            return _t.load(&_store_a[_r]);
        }

    private:
        using base_t          = typename T::base_t;
        using bases_t         = typename T::bases_t;
        using store_t         = std::vector<internal_t, boost::alignment::aligned_allocator<internal_t, T::alignment>>;
        using difference_type = typename store_t::difference_type;

        const std::size_t   _r;
        const std::size_t   _r2;
        const std::size_t   _r2_plus;
        source_t            _source;
        const std::size_t   _length;
        const std::size_t   _length_minus;
        const std::size_t   _length_minus_r;
        store_t             _store_a;
        store_t             _store_b;
        store_t             _store_tmp;
        T                   _t;

        void inner_loop(const calc_t* local_base_i, std::size_t idx_i, std::size_t idx_j_min, std::size_t idx_j_max, std::size_t store_delta) {
            std::size_t store_pos = store_delta;

            internal_t current_value   = _t.infinity();
            internal_t last_value_here = _t.infinity();  // will never be used
            internal_t last_value_next = _t.load(&_store_a[store_pos]);

            std::fill(
                _store_b.begin(),
                _store_b.begin() + static_cast<difference_type>(store_pos),
                _t.infinity()
            );

            for (std::size_t idx_j = idx_j_min; idx_j <= idx_j_max; ++idx_j) {
                internal_t a = _t.convert_single(local_base_i[idx_i]);
                internal_t b = _t.load(&_store_tmp[idx_j]);
                internal_t cost = dist(a, b);

                last_value_here = last_value_next;
                if (store_pos < _r2) {
                    last_value_next = _t.load(&_store_a[store_pos + 1]);
                } else {
                    last_value_next = _t.infinity();
                }

                // rename variables (let's hope the compiler optimizes that)
                internal_t dtw_insert = last_value_next;
                internal_t dtw_delete = current_value;  // value from last round
                internal_t dtw_match  = last_value_here;

                current_value = cost + _t.min3(dtw_insert, dtw_delete, dtw_match);
                _t.store(&_store_b[store_pos], current_value);

                ++store_pos;
            }

            std::fill(
                _store_b.begin() + static_cast<difference_type>(store_pos),
                _store_b.end(),
                _t.infinity()
            );

            std::swap(_store_a, _store_b);
        }

        void load_to_tmp(indices_t j0) {
            bases_t local_bases_j = _t.get_bases(_source, _length, j0);
            for (std::size_t idx = 0; idx < _length; ++idx) {
                _t.store(&_store_tmp[idx], _t.convert_multiple(local_bases_j, idx));
            }
        }
};

struct dtw_impl_simple {
    static constexpr std::size_t n         = 1;
    static constexpr std::size_t alignment = n * 8;

    using internal_t = calc_t;
    using indices_t  = std::size_t;
    using source_t   = const calc_t*;
    using base_t     = const calc_t*;
    using bases_t    = const calc_t*;

    inline internal_t convert_single(calc_t x) {
        return x;
    }

    inline internal_t convert_multiple(bases_t bases, std::size_t idx) {
        return static_cast<calc_t>(bases[idx]);
    }

    inline base_t get_base(source_t source, std::size_t length, std::size_t i) {
        return source + (i * length);
    }

    inline bases_t get_bases(source_t source, std::size_t length, indices_t j0) {
        return source + (j0 * length);
    }

    inline internal_t load(internal_t* ptr) {
        return *ptr;
    }

    inline internal_t min3(internal_t a, internal_t b, internal_t c) {
        return std::min(a, std::min(b, c));
    }

    inline internal_t infinity() {
        return std::numeric_limits<calc_t>::infinity();
    }

    inline void store(internal_t* ptr, internal_t data) {
        *ptr = data;
    }

    inline internal_t zero() {
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
    using source_t   = const calc_t*;
    using base_t     = const calc_t*;
    using bases_t    = std::array<const calc_t*, n>;

    inline internal_t convert_single(calc_t x) {
        return simdpp::make_float(x);
    }

    inline internal_t convert_multiple(bases_t bases, std::size_t idx) {
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

    inline base_t get_base(source_t source, std::size_t length, std::size_t i) {
        return source + (i * length);
    }

    inline internal_t load(internal_t* ptr) {
        return simdpp::load(ptr);
    }

    inline internal_t min3(internal_t a, internal_t b, internal_t c) {
        return simdpp::min(a, simdpp::min(b, c));
    }

    inline internal_t infinity() {
        return simdpp::make_float(std::numeric_limits<double>::infinity());
    }

    inline void store(internal_t* ptr, internal_t data) {
        simdpp::store(ptr, data);
    }

    inline internal_t zero() {
        return simdpp::make_float(0.0);
    }
};

struct dtw_impl_vectorized_linear : dtw_impl_vectorized {
    using indices_t = std::size_t;

    inline bases_t get_bases(source_t source, std::size_t length, indices_t j0) {
        return {{
            (source + ((j0 +  0) * length)),
            (source + ((j0 +  1) * length)),
            (source + ((j0 +  2) * length)),
            (source + ((j0 +  3) * length)),
            (source + ((j0 +  4) * length)),
            (source + ((j0 +  5) * length)),
            (source + ((j0 +  6) * length)),
            (source + ((j0 +  7) * length)),
            (source + ((j0 +  8) * length)),
            (source + ((j0 +  9) * length)),
            (source + ((j0 + 10) * length)),
            (source + ((j0 + 11) * length)),
            (source + ((j0 + 12) * length)),
            (source + ((j0 + 13) * length)),
            (source + ((j0 + 14) * length)),
            (source + ((j0 + 15) * length))
        }};
    }
};

struct dtw_impl_vectorized_shuffled : dtw_impl_vectorized {
    using indices_t = const std::array<std::size_t, n>&;

    inline bases_t get_bases(source_t source, std::size_t length, indices_t j0) {
        return {{
            (source + (std::get< 0>(j0) * length)),
            (source + (std::get< 1>(j0) * length)),
            (source + (std::get< 2>(j0) * length)),
            (source + (std::get< 3>(j0) * length)),
            (source + (std::get< 4>(j0) * length)),
            (source + (std::get< 5>(j0) * length)),
            (source + (std::get< 6>(j0) * length)),
            (source + (std::get< 7>(j0) * length)),
            (source + (std::get< 8>(j0) * length)),
            (source + (std::get< 9>(j0) * length)),
            (source + (std::get<10>(j0) * length)),
            (source + (std::get<11>(j0) * length)),
            (source + (std::get<12>(j0) * length)),
            (source + (std::get<13>(j0) * length)),
            (source + (std::get<14>(j0) * length)),
            (source + (std::get<15>(j0) * length)),
        }};
    }
};

using dtw_simple              = dtw_generic<dtw_impl_simple>;
using dtw_vectorized_linear   = dtw_generic<dtw_impl_vectorized_linear>;
using dtw_vectorized_shuffled = dtw_generic<dtw_impl_vectorized_shuffled>;
