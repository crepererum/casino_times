#include <cstdint>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
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
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "dtw.hpp"
#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;


int main(int argc, char** argv) {
    std::locale loc;
    if (gen_locale(loc)) {
        return 1;
    }

    std::string fname_binary;
    std::string fname_map;
    std::vector<std::string> query;
    std::size_t r;
    std::size_t limit;
    year_t begin;
    year_t end;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("begin", po::value(&begin), "first year that should be queried (starts at 0), defaults to 0")
        ("end", po::value(&end), "end of the year range (last year + 1), defaults to ylength")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("r", po::value(&r)->required(), "radius of Sakoe-Chiba Band")
        ("limit", po::value(&limit), "number of ngrams to look for")
        ("query", po::value(&query)->required()->multitoken(), "query ngram")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "query_dtw_simple")) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }
    if (!vm.count("begin")) {
        begin = 0;
    }
    if (!vm.count("end")) {
        end = ylength;
    }
    if (end <= begin) {
        std::cerr << "end has to be greater than begin!" << std::endl;
        return 1;
    }
    if (end > ylength) {
        std::cerr << "end has to be at max ylength!" << std::endl;
        return 1;
    }
    if (r > ((end - begin) >> 1)) {
        std::cerr << "r has to be <= (end - begin)/2!" << std::endl;
        return 1;
    }

    idx_ngram_map_t idxmap;
    ngram_idx_map_t ngmap;
    std::tie(idxmap, ngmap) = parse_map_file(fname_map);
    std::size_t n = ngmap.size();

    std::vector<std::size_t> i;
    for (const auto& q : query) {
        auto query_utf32 = boost::locale::conv::utf_to_utf<char32_t>(q);
        auto ngram_it = ngmap.find(query_utf32);
        if (ngram_it == ngmap.end()) {
            std::cerr << "unkown ngram: \"" << q << "\"" << std::endl;
            return 1;
        }
        i.emplace_back(ngram_it->second);
    }

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    std::vector<std::pair<std::size_t, float>> distances(n);
    for (std::size_t j = 0; j < n; ++j) {
        distances[j] = std::make_pair(j, 0.0);
    }

    const std::size_t n_over = n % static_cast<std::size_t>(dtw_vectorized_linear::n);
    const std::size_t n_good = n - n_over;

    dtw_vectorized_linear mydtw_vectorized(base, ylength, begin, end, r);
    dtw_simple mydtw_simple(base, ylength, begin, end, r);

    // this loop order is faster than the opposite way
    for (std::size_t j = 0; j < n_good; j += dtw_vectorized_linear::n) {
        for (const std::size_t i_part : i) {
            auto v_results = mydtw_vectorized.calc(i_part, j);

            std::array<float, dtw_vectorized_linear::n> d_results;
            simdpp::store(&d_results, v_results);

            for (std::size_t idx = 0; idx < dtw_vectorized_linear::n; ++idx) {
                // d_results are squared at this point!
                distances[j + idx].second += d_results[idx];
            }
        }
    }

    // ... dito ;)
    for (std::size_t j = n_good; j < n; ++j) {
        for (const std::size_t i_part : i) {
            // result is squared at this point!
            distances[j].second += mydtw_simple.calc(i_part, j);
        }
    }

    std::size_t usable_limit;
    if (vm.count("limit")) {
        std::cout << "limit: " << limit << std::endl;
        usable_limit = std::min(limit, n);
        std::partial_sort(
            distances.begin(),
            distances.begin() + static_cast<std::iterator_traits<decltype(distances.begin())>::difference_type>(usable_limit),
            distances.end(),
            [](const auto& a, const auto& b){
                return a.second < b.second;
            }
        );
    } else {
        std::cout << "limit (autodetection): " << std::flush;

        std::sort(
            distances.begin(),
            distances.end(),
            [](const auto& a, const auto& b){
                return a.second < b.second;
            }
        );

        std::vector<float> dn(n - 1);
        float d1 = distances[1].second;
        for (std::size_t idx = 0; idx < n - 1; ++idx) {
            float d_idx_norm = (distances[idx + 1].second - d1);
            dn[idx] = d_idx_norm;
        }

        std::vector<float> grad1_norm = gradient(dn);
        for (std::size_t idx = 0; idx < n; ++idx) {
            grad1_norm[idx] /= dn[idx];
        }

        std::vector<float> grad2 = gradient(grad1_norm);

        using dtype = std::iterator_traits<decltype(grad2.begin())>::difference_type;
        auto it = std::max_element(
            grad2.begin() + static_cast<dtype>(1),
            grad2.begin() + static_cast<dtype>(n / 2)
        );
        usable_limit = limit = 1 + static_cast<std::size_t>(std::distance(grad2.begin(), it));

        std::cout << limit << std::endl;
    }

    constexpr std::size_t colw0 = 5;
    constexpr std::size_t colw1 = 20;
    constexpr std::size_t colw2 = 10;
    constexpr std::size_t colw3 = 10;
    std::cout
        << "| "
        << std::setw(colw0) << "rank"
        << " | "
        << std::setw(colw1) << "ngram"
        << " | "
        << std::setw(colw2) << "id"
        << " | "
        << std::setw(colw3) << "distance"
        << " |"
        << std::endl;
    std::cout
        << "|-"
        << std::string(colw0, '-')
        << "-|-"
        << std::string(colw1, '-')
        << "-|-"
        << std::string(colw2, '-')
        << "-|-"
        << std::string(colw3, '-')
        << "-|"
        << std::endl;
    for (std::size_t j = 0; j < usable_limit; ++j) {
        std::cout
            << "| "
            << std::setw(colw0) << j
            << " | "
            << std::setw(colw1) << boost::locale::conv::utf_to_utf<char>(idxmap[distances[j].first])
            << " | "
            << std::setw(colw2) << distances[j].first
            << " | "
            << std::setw(colw3) << (std::sqrt(distances[j].second / static_cast<float>(end - begin)))
            << " |"
            << std::endl;
    }
}
