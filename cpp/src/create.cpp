#include <cstdint>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_out0;
    std::string fname_out1;
    std::string fname_mapin;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary0", po::value(&fname_out0)->required(), "output binary file for var0")
        ("binary1", po::value(&fname_out1)->required(), "output binary file for var1")
        ("map", po::value(&fname_mapin)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "create") != 0) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }

    idx_ngram_map_t idxmap;
    ngram_idx_map_t ngmap;
    std::tie(idxmap, ngmap) = parse_map_file(fname_mapin);
    std::size_t n = ngmap.size();

    std::size_t fsize = n * ylength * sizeof(var_t);
    auto output0 = open_raw_file(fname_out0, fsize, true, true);
    auto output1 = open_raw_file(fname_out1, fsize, true, true);
    if (!output0.is_open()) {
        std::cerr << "cannot write output file for var0" << std::endl;
        return 1;
    }
    if (!output1.is_open()) {
        std::cerr << "cannot write output file for var1" << std::endl;
        return 1;
    }
}
