#include <cstdint>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

constexpr std::size_t dtw_resolution = 16;
static_assert(is_power_of_2(dtw_resolution), "DTW resolution has to be a power of 2!");
static_assert(dtw_resolution >= 2, "DTW resolution has to be at least 2!");

using calc_t = double;
static_assert(sizeof(double) == 8, "double isn't 64bit :(");

std::vector<calc_t> get_downsampled(const calc_t* base, year_t ylength, std::size_t i) {
    std::vector<calc_t> result(dtw_resolution, 0.0);
    std::size_t rate = ylength / dtw_resolution;
    std::size_t target = 0;
    const calc_t* local_base = base + (i * ylength);

    for (std::size_t idx = 0; idx < ylength; ++idx) {
        result[target] += static_cast<calc_t>(local_base[idx]);

        if (idx >= rate * target) {
            ++target;
        }
    }

    calc_t norm_factor = static_cast<calc_t>(dtw_resolution) / static_cast<calc_t>(ylength);
    for (auto& x : result) {
        x *= norm_factor;
    }

    return result;
}

calc_t dist(calc_t a, calc_t b) {
    calc_t d = a - b;
    return d * d;
}

class dtw {
    public:
        dtw(const calc_t* base, year_t ylength, std::size_t r) : _r(r), _r_plus(_r + 1), _r2(_r << 1), _r2_plus(_r2 + 1), _base(base), _ylength(ylength), _ylength_plus(_ylength + 1), _ylength_plus_minus_r(_ylength_plus - _r), _store_a(_r2_plus), _store_b(_r2_plus) {}

        calc_t calc(std::size_t i, std::size_t j) {
            const calc_t* local_base_i = _base + (i * _ylength);
            const calc_t* local_base_j = _base + (j * _ylength);

            std::fill(_store_a.begin(), _store_a.end(), std::numeric_limits<calc_t>::infinity());
            _store_a[_r] = 0.0;

            // remember: r <= ylength/2

            for (std::size_t idx_i = 1; idx_i < _r_plus; ++idx_i) {
                std::size_t idx_j_min = 1;
                std::size_t idx_j_max = idx_i + _r;

                inner_loop(local_base_i, local_base_j, idx_i, idx_j_min, idx_j_max, _r + 1 - idx_i);
            }

            for (std::size_t idx_i = _r_plus; idx_i < _ylength_plus_minus_r; ++idx_i) {
                std::size_t idx_j_min = idx_i - _r;
                std::size_t idx_j_max = idx_i + _r;

                inner_loop(local_base_i, local_base_j, idx_i, idx_j_min, idx_j_max, 0);
            }

            for (std::size_t idx_i = _ylength_plus_minus_r; idx_i < _ylength_plus; ++idx_i) {
                std::size_t idx_j_min = idx_i - _r;
                std::size_t idx_j_max = _ylength;  // = _ylength_plus - 1;

                inner_loop(local_base_i, local_base_j, idx_i, idx_j_min, idx_j_max, 0);
            }

            return _store_a[_r];
        }

    private:
        const std::size_t _r;
        const std::size_t _r_plus;
        const std::size_t _r2;
        const std::size_t _r2_plus;
        const calc_t* const _base;
        const year_t _ylength;
        const year_t _ylength_plus;
        const std::size_t _ylength_plus_minus_r;
        std::vector<calc_t> _store_a;
        std::vector<calc_t> _store_b;

        void inner_loop(const calc_t* local_base_i, const calc_t* local_base_j, std::size_t idx_i, std::size_t idx_j_min, std::size_t idx_j_max, std::size_t store_delta) {
            std::size_t store_pos = store_delta;
            calc_t current_value = std::numeric_limits<calc_t>::infinity();

            std::fill(
                _store_b.begin(),
                _store_b.begin() + static_cast<std::iterator_traits<decltype(_store_b.begin())>::difference_type>(store_pos),
                std::numeric_limits<calc_t>::infinity()
            );

            for (std::size_t idx_j = idx_j_min; idx_j <= idx_j_max; ++idx_j) {
                calc_t cost = dist(
                    static_cast<calc_t>(local_base_i[idx_i]),
                    static_cast<calc_t>(local_base_j[idx_j])
                );

                calc_t dtw_insert = std::numeric_limits<calc_t>::infinity();
                if (store_pos < _r2) {
                    // XXX: this is only illegal during the last round, might consider unrolling
                    dtw_insert = _store_a[store_pos + 1];
                }
                calc_t dtw_delete = current_value;  // value from last round
                calc_t dtw_match  = _store_a[store_pos];

                current_value = cost + std::min(dtw_insert, std::min(dtw_delete, dtw_match));
                _store_b[store_pos] = current_value;

                ++store_pos;
            }

            std::fill(
                _store_b.begin() + static_cast<std::iterator_traits<decltype(_store_b.begin())>::difference_type>(store_pos),
                _store_b.end(),
                std::numeric_limits<calc_t>::infinity()
            );

            std::swap(_store_a, _store_b);
        }
};


int main(int argc, char** argv) {
    std::locale loc;
    if (gen_locale(loc)) {
        return 1;
    }

    std::string fname_binary;
    std::string fname_map;
    std::string query;
    std::size_t r;
    std::size_t limit;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("r", po::value(&r)->required(), "radius of Sakoe-Chiba Band")
        ("limit", po::value(&limit)->required(), "number of ngrams to look for")
        ("query", po::value(&query)->required(), "query ngram")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "query_dtw")) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }
    if (ylength < dtw_resolution) {
        std::cerr << "ylength has to be at least " << dtw_resolution << "!" << std::endl;
        return 1;
    }
    if (r > (ylength >> 1)) {
        std::cerr << "r has to be <= ylength/2!" << std::endl;
        return 1;
    }

    idx_ngram_map_t idxmap;
    ngram_idx_map_t ngmap;
    std::tie(idxmap, ngmap) = parse_map_file(fname_map);
    std::size_t n = ngmap.size();

    auto query_utf32 = boost::locale::conv::utf_to_utf<char32_t>(query);
    auto ngram_it = ngmap.find(query_utf32);
    if (ngram_it == ngmap.end()) {
        std::cerr << "unkown ngram" << std::endl;
        return 1;
    }
    std::size_t i = ngram_it->second;

    boost::iostreams::mapped_file_params params;
    params.path   = fname_binary;
    params.flags  = boost::iostreams::mapped_file::mapmode::readonly;
    params.length = static_cast<std::size_t>(n * ylength * sizeof(calc_t));
    params.offset = 0;
    boost::iostreams::mapped_file input(params);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    std::size_t usable_limit = std::min(limit, n);
    std::vector<std::pair<std::size_t, calc_t>> distances(n);
    dtw mydtw(base, ylength, r);
    for (std::size_t j = 0; j < n; ++j) {
        distances[j] = std::make_pair(j, mydtw.calc(i, j));
    }
    std::partial_sort(
        distances.begin(),
        distances.begin() + static_cast<std::iterator_traits<decltype(distances.begin())>::difference_type>(usable_limit),
        distances.end(),
        [](const auto& a, const auto& b){
            return a.second < b.second;
        }
    );
    for (std::size_t j = 0; j < usable_limit; ++j) {
        std::cout
            << boost::locale::conv::utf_to_utf<char>(idxmap[distances[j].first])
            << " : "
            << distances[j].second
            << std::endl;
    }
}
