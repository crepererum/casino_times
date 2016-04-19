#include <cstdint>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "dtw.hpp"
#include "dtw_index.hpp"
#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_binary;
    std::string fname_map;
    std::string fname_index;
    std::size_t index_size;
    std::size_t r;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("r", po::value(&r)->required(), "radius of Sakoe-Chiba Band")
        ("index", po::value(&fname_index)->required(), "index file")
        ("size", po::value(&index_size)->required(), "size of the index file (in bytes)")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "index_dtw")) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }
    if (ylength < dtw_index_resolution) {
        std::cerr << "ylength has to be at least " << dtw_index_resolution << "!" << std::endl;
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

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    std::cout << "open index file..." << std::endl;
    boost::interprocess::managed_mapped_file findex(boost::interprocess::create_only, fname_index.c_str(), index_size);
    std::cout << "done" << std::endl;

    std::cout << "do downsampling..." << std::endl;
    allocator_point_t alloc_point(findex.get_segment_manager());
    downstorage_t* down = findex.construct<downstorage_t>("down")(n, alloc_point);
    // XXX: vectorize
    for (std::size_t j = 0; j < n; ++j) {
        if (j % 100000 == 0) {
            std::cout << "  " << j << "/" << n << std::endl;
        }

        get_downsampled_data::f(base + (j * ylength), ylength, (*down)[j]);
    }
    std::cout << "done" << std::endl;

    // DO:
    //   create a collection of all points first and then create a index out of the entire collection (packing algorithm)
    // DON'T:
    //   create empty index and insert points individually

    std::cout << "create cloud..." << std::endl;
    std::vector<node_t> cloud(n);
    // XXX: vectorize
    for (std::size_t j = 0; j < n; ++j) {
        if (j % 100000 == 0) {
            std::cout << "  " << j << "/" << n << std::endl;
        }

        cloud[j].second = j;

        auto lu = get_lu(base, ylength, j, r);
        get_downsampled_l::f(lu.first.data(), ylength, cloud[j].first.min_corner());
        get_downsampled_u::f(lu.second.data(), ylength, cloud[j].first.max_corner());

    }
    std::cout << "done" << std::endl;

    std::cout << "create index..." << std::endl;
    allocator_node_t alloc_node(findex.get_segment_manager());
    findex.construct<tree_t>("rtree")(cloud, params_t(), indexable_t(), equal_to_t(), alloc_node);
    std::cout << "done" << std::endl;

    std::cout << "Free memory:" << (findex.get_free_memory() >> 10) << "k of " << (findex.get_size() >> 10) << "k" << std::endl;
}
