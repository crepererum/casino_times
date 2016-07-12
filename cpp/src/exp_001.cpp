#include <cstdint>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "dtw.hpp"
#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

namespace std {
    template<>
    struct hash<std::pair<std::size_t, std::size_t>> {
        size_t operator()(const std::pair<std::size_t, std::size_t>& obj) const {
            return obj.first ^ obj.second;
        }
    };
}


int main(int argc, char** argv) {
    std::locale loc;
    if (gen_locale(loc)) {
        return 1;
    }

    std::string fname_binary;
    std::string fname_map;
    std::string fname_dot;
    std::string query;
    std::size_t r;
    std::size_t limit;
    std::size_t howmany;
    year_t begin;
    year_t end;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("dot", po::value(&fname_dot)->required(), "output dot file")
        ("begin", po::value(&begin), "first year that should be queried (starts at 0), defaults to 0")
        ("end", po::value(&end), "end of the year range (last year + 1), defaults to ylength")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("r", po::value(&r)->required(), "radius of Sakoe-Chiba Band")
        ("limit", po::value(&limit)->required(), "number of ngrams to look for")
        ("query", po::value(&query)->required(), "query ngram")
        ("howmany", po::value(&howmany)->required(), "how many")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "exp_001")) {
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
    if (r > static_cast<std::size_t>((end - begin) / 2)) {
        std::cerr << "r has to be <= (end - begin)/2!" << std::endl;
        return 1;
    }

    idx_ngram_map_t idxmap;
    ngram_idx_map_t ngmap;
    std::tie(idxmap, ngmap) = parse_map_file(fname_map);
    std::size_t n = ngmap.size();

    auto query_utf32 = boost::locale::conv::utf_to_utf<char32_t>(query);
    auto ngram_it = ngmap.find(query_utf32);
    if (ngram_it == ngmap.end()) {
        std::cerr << "unkown ngram: \"" << query << "\"" << std::endl;
        return 1;
    }
    std::size_t i_0 = ngram_it->second;

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    const std::size_t n_over = n % static_cast<std::size_t>(dtw_vectorized_linear::n);
    const std::size_t n_good = n - n_over;
    const std::size_t usable_limit = std::min(limit, n);

    std::unordered_map<std::pair<std::size_t, std::size_t>, float> result;
    std::vector<std::pair<std::size_t, float>> distances(n);
    dtw_vectorized_linear mydtw_vectorized(base, ylength, begin, end, r);
    dtw_simple mydtw_simple(base, ylength, begin, end, r);
    std::unordered_set<std::size_t> done{};
    std::unordered_set<std::size_t> seen{i_0};
    std::unordered_map<std::size_t, float> backlog{{i_0, 0.f}};

    std::cout << "Work: " << std::flush;
    while (!backlog.empty() && done.size() < howmany) {
        std::size_t i = std::min_element(backlog.begin(), backlog.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        })->first;

        for (std::size_t j = 0; j < n_good; j += dtw_vectorized_linear::n) {
            auto v_results = mydtw_vectorized.calc(i, j);

            std::array<float, dtw_vectorized_linear::n> d_results;
            simdpp::store(&d_results, v_results);

            for (std::size_t idx = 0; idx < dtw_vectorized_linear::n; ++idx) {
                // d_results are squared at this point!
                distances[j + idx] = std::make_pair(j + idx, d_results[idx]);
            }
        }

        for (std::size_t j = n_good; j < n; ++j) {
            // result is squared at this point!
            distances[j] = std::make_pair(j, mydtw_simple.calc(i, j));
        }

        std::partial_sort(
            distances.begin(),
            distances.begin() + static_cast<std::iterator_traits<decltype(distances.begin())>::difference_type>(usable_limit),
            distances.end(),
            [](const auto& a, const auto& b){
                return a.second < b.second;
            }
        );

        float norm = 1.f / std::sqrt(distances[1].second);
        for (std::size_t j = 0; j < usable_limit; ++j) {
            std::size_t j_real = distances[j].first;
            float dist = std::sqrt(distances[j].second) * norm;

            std::size_t a = i;
            std::size_t b = j_real;
            if (b > a) {
                std::swap(a, b);
            }

            result.emplace(std::make_pair(std::make_pair(a, b), dist));

            auto it1 = backlog.find(j_real);
            if (it1 != backlog.end()) {
                it1->second = std::min(it1->second, dist);
            } else {
                if (done.find(j_real) == done.end()) {
                    backlog.emplace(std::make_pair(j_real, dist));
                    seen.emplace(j_real);
                }
            }
        }

        backlog.erase(i);
        done.emplace(i);
        std::cout << "." << std::flush;
    }
    std::cout << "done" << std::endl;

    std::ofstream out(fname_dot);

    out << "graph cloud {" << std::endl;

    out << "  overlap = false;" << std::endl;
    out << std::endl;

    for (const auto& i : seen) {
        out << "  node"
            << i
            << " [label=\""
            << boost::locale::conv::utf_to_utf<char>(idxmap[i])
            << "\", shape=circle];"
            << std::endl;
    }
    out << std::endl;

    for (const auto& kv : result) {
        if (kv.first.first != kv.first.second) {
            out << "  node"
                << kv.first.first
                << " -- "
                << " node"
                << kv.first.second
                << ";"
                << std::endl;
        }
    }
    out << "}" << std::endl;
}
