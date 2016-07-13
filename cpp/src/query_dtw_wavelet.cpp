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
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "dtw.hpp"
#include "parser.hpp"
#include "tracer.hpp"
#include "utils.hpp"
#include "wavelet_tree.hpp"


struct tracer_impl_dtw : tracer_impl_limiter {
    std::unordered_set<std::size_t> candidates;

    tracer_impl_dtw(std::size_t ylength, std::size_t maxdepth, std::size_t begin, std::size_t end, float minweight)
        : tracer_impl_limiter(power_of_2(ylength), maxdepth, begin, end, minweight) {}

    void found_superroot(const superroot_ptr_t& s, float) {
        candidates.emplace(s->i);
    }
};

namespace po = boost::program_options;


int main(int argc, char** argv) {
    std::locale loc;
    if (gen_locale(loc)) {
        return 1;
    }

    std::string fname_binary;
    std::string fname_map;
    std::string fname_wavelet;
    std::string query;
    std::size_t r;
    std::size_t limit;
    std::size_t maxdepth;
    float minweight = 0.f;
    year_t begin;
    year_t end;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("wavelet", po::value(&fname_wavelet)->required(), "wavelet data file")
        ("begin", po::value(&begin), "first year that should be queried (starts at 0), defaults to 0")
        ("end", po::value(&end), "end of the year range (last year + 1), defaults to ylength")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("r", po::value(&r)->required(), "radius of Sakoe-Chiba Band")
        ("limit", po::value(&limit)->required(), "number of ngrams to look for")
        ("query", po::value(&query)->required(), "query ngram")
        ("maxdepth", po::value(&maxdepth)->required(), "maximum search depth")
        ("minweight", po::value(&minweight)->required(), "minimum weight during trace-up")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "query_dtw_wavelet")) {
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

    auto query_utf32 = boost::locale::conv::utf_to_utf<char32_t>(query);
    auto ngram_it = ngmap.find(query_utf32);
    if (ngram_it == ngmap.end()) {
        std::cerr << "unkown ngram: \"" << query << "\"" << std::endl;
        return 1;
    }
    std::size_t i = ngram_it->second;

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    std::cout << "open index file..." << std::endl;
    auto findex = std::make_shared<boost::interprocess::managed_mapped_file>(
        boost::interprocess::open_only,
        fname_wavelet.c_str()
    );
    index_stored_t index(findex, idxmap.size());
    std::cout << "done" << std::endl;

    tracer<tracer_impl_dtw> t{
        &index,
        {
            ylength,
            maxdepth,
            begin,
            end,
            minweight
        }
    };
    t(i);

    const std::size_t n_new = t.t().candidates.size();
    std::cout << "#dtw=" << n_new << std::endl;

    std::vector<std::size_t> candidates_sorted(t.t().candidates.begin(), t.t().candidates.end());
    std::sort(candidates_sorted.begin(), candidates_sorted.end());  // sort to speed-up memory ops

    dtw_vectorized_shuffled mydtw_vectorized(base, ylength, begin, end, r);
    dtw_simple mydtw_simple(base, ylength, begin, end, r);
    const std::size_t n_over = n_new % static_cast<std::size_t>(dtw_vectorized_linear::n);
    const std::size_t n_good = n_new - n_over;
    std::vector<std::pair<std::size_t, float>> distances(n_new);

    std::array<std::size_t, dtw_vectorized_shuffled::n> vindices;
    for (std::size_t j_base = 0; j_base < n_good; j_base += dtw_vectorized_linear::n) {
        for (std::size_t idx = 0; idx < dtw_vectorized_shuffled::n; ++idx) {
            vindices[idx] = candidates_sorted[j_base + idx];
        }

        auto v_results = mydtw_vectorized.calc(i, vindices);

        std::array<float, dtw_vectorized_linear::n> d_results;
        simdpp::store(&d_results, v_results);

        for (std::size_t idx = 0; idx < dtw_vectorized_linear::n; ++idx) {
            // d_results are squared at this point!
            distances[j_base + idx] = std::make_pair(vindices[idx], d_results[idx]);
        }
    }

    for (std::size_t j_base = n_good; j_base < n_new; ++j_base) {
        std::size_t j = candidates_sorted[j_base];

        // result is squared at this point!
        distances[j_base] = std::make_pair(j, mydtw_simple.calc(i, j));
    }

    const std::size_t usable_limit = std::min(limit, n_new);
    std::partial_sort(
        distances.begin(),
        distances.begin() + static_cast<std::iterator_traits<decltype(distances.begin())>::difference_type>(usable_limit),
        distances.end(),
        [](const auto& a, const auto& b){
            return a.second < b.second;
        }
    );

    constexpr std::size_t colw0 = 20;
    constexpr std::size_t colw1 = 10;
    constexpr std::size_t colw2 = 10;
    std::cout
        << "| "
        << std::setw(colw0) << "ngram"
        << " | "
        << std::setw(colw1) << "id"
        << " | "
        << std::setw(colw2) << "distance"
        << " |"
        << std::endl;
    std::cout
        << "|-"
        << std::string(colw0, '-')
        << "-|-"
        << std::string(colw1, '-')
        << "-|-"
        << std::string(colw2, '-')
        << "-|"
        << std::endl;
    for (std::size_t j = 0; j < usable_limit; ++j) {
        std::cout
            << "| "
            << std::setw(colw0) << boost::locale::conv::utf_to_utf<char>(idxmap[distances[j].first])
            << " | "
            << std::setw(colw1) << distances[j].first
            << " | "
            << std::setw(colw2) << (std::sqrt(distances[j].second / static_cast<float>(end - begin)))
            << " |"
            << std::endl;
    }
}
