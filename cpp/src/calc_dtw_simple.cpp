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
    std::string fname_binary;
    std::string fname_map;
    std::string fname_out;
    std::string query;
    std::size_t r;
    year_t begin;
    year_t end;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary",  po::value(&fname_binary)->required(), "input binary file for var")
        ("map",     po::value(&fname_map)->required(),    "ngram map file to read")
        ("begin", po::value(&begin), "first year that should be queried (starts at 0), defaults to 0")
        ("end", po::value(&end), "end of the year range (last year + 1), defaults to ylength")
        ("ylength", po::value(&ylength)->required(),      "number of years to store")
        ("r",       po::value(&r)->required(),            "radius of Sakoe-Chiba Band")
        ("query",   po::value(&query)->required(),        "query ngram")
        ("output",  po::value(&fname_out)->required(),    "output data")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "calc_dtw_simple") != 0) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }
    if (vm.count("begin") == 0u) {
        begin = 0;
    }
    if (vm.count("end") == 0u) {
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
        std::cerr << "unkown ngram" << std::endl;
        return 1;
    }
    std::size_t i = ngram_it->second;

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base_in = reinterpret_cast<const calc_t*>(input.const_data());

    auto output = open_raw_file(fname_out, n * sizeof(calc_t), true, true);
    if (!output.is_open()) {
        std::cerr << "cannot open output file" << std::endl;
        return 1;
    }
    auto base_out = reinterpret_cast<calc_t*>(output.data());

    dtw_simple mydtw_simple(base_in, ylength, begin, end, r);
    for (std::size_t j = 0; j < n; ++j) {
        base_out[j] = (std::sqrt(static_cast<calc_t>(mydtw_simple.calc(i, j)) / static_cast<calc_t>(end - begin)));
    }
}
