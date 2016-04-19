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
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::locale loc;
    if (gen_locale(loc)) {
        return 1;
    }

    std::string fname_in;
    std::string fname_out0;
    std::string fname_out1;
    std::string fname_mapin;
    std::string fname_trans;
    year_t ystart;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("file", po::value(&fname_in)->required(), "input ngram file")
        ("binary0", po::value(&fname_out0)->required(), "output binary file for var0")
        ("binary1", po::value(&fname_out1)->required(), "output binary file for var1")
        ("map", po::value(&fname_mapin)->required(), "ngram map file to read")
        ("trans", po::value(&fname_trans), "transition map (e.g. for stemming)")
        ("ystart", po::value(&ystart)->required(), "first year to store")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "store")) {
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

    if (vm.count("trans")) {
        auto t = parse_trans_file(fname_trans);

        ngram_idx_map_t tmp = ngmap;
        ngmap = ngram_idx_map_t();

        for (const auto& kv : t) {
            auto it = tmp.find(kv.second);

            // there are cases where the stemmer messed up unicode chars, so be careful
            if (it != tmp.end()) {
                ngmap.insert(std::make_pair(kv.first, it->second));
            }
        }
    }

    boost::iostreams::mapped_file input(fname_in, boost::iostreams::mapped_file::mapmode::readonly);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto fit = input.const_data();
    auto fend = fit + input.size();

    std::size_t fsize = n * ylength * sizeof(var_t);
    auto output0 = open_raw_file(fname_out0, fsize, true, false);
    auto output1 = open_raw_file(fname_out1, fsize, true, false);
    if (!output0.is_open()) {
        std::cerr << "cannot write output file for var0" << std::endl;
        return 1;
    }
    if (!output1.is_open()) {
        std::cerr << "cannot write output file for var1" << std::endl;
        return 1;
    }
    auto base0 = reinterpret_cast<var_t*>(output0.data());
    auto base1 = reinterpret_cast<var_t*>(output1.data());

    year_t yend = ystart + ylength;
    std::size_t n_entries_total = 0;
    std::size_t n_entires_written = 0;
    while (fit != fend) {
        auto entry = parse_line_to_entry(fit, fend);
        auto ngram = normalize(entry.ngram, loc);
        auto it = ngmap.find(ngram);
        if (entry.year >= ystart && entry.year < yend && it != ngmap.end()) {
            std::size_t y = entry.year - ystart;
            std::size_t i = it->second;
            std::size_t offset = i * ylength + y;
            base0[offset] += entry.var0;
            base1[offset] += entry.var1;
            ++n_entires_written;
        }
        ++n_entries_total;
    }

    std::cout << "#Entries(total)=" << n_entries_total << std::endl
        << "#Entries(written)=" << n_entires_written << std::endl;
}
