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
    if (po_fill_vm(desc, vm, argc, argv, "create")) {
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


    boost::iostreams::mapped_file_params params0;
    boost::iostreams::mapped_file_params params1;
    params0.path = fname_out0;
    params1.path = fname_out1;
    params0.flags = params1.flags = boost::iostreams::mapped_file::mapmode::readwrite;
    params0.new_file_size = params1.new_file_size = static_cast<boost::iostreams::stream_offset>(n * ylength * sizeof(var_t));
    params0.length = params1.length = static_cast<std::size_t>(n * ylength * sizeof(var_t));
    params0.offset = params1.offset = 0;
    boost::iostreams::mapped_file output0(params0);
    boost::iostreams::mapped_file output1(params1);
    if (!output0.is_open()) {
        std::cerr << "cannot write output file for var0" << std::endl;
        return 1;
    }
    if (!output1.is_open()) {
        std::cerr << "cannot write output file for var1" << std::endl;
        return 1;
    }
}
