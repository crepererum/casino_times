#include <cstdint>

#include <algorithm>
#include <iostream>
#include <string>

#include <boost/iostreams/device/mapped_file.hpp>
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
    std::string fname_mapout;
    auto desc = po_create_desc();
    desc.add_options()
        ("file", po::value(&fname_in)->required(), "input ngram file")
        ("map", po::value(&fname_mapout)->required(), "ngram map file to write")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "scan")) {
        return 1;
    }

    boost::iostreams::mapped_file input(fname_in, boost::iostreams::mapped_file::mapmode::readonly);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto fit = input.const_data();
    auto fend = fit + input.size();

    ngram_set_t ngrams;
    year_t year_min = 2000;
    year_t year_max = 2000;
    var_t var_max = 0;
    while (fit != fend) {
        auto entry = parse_line_to_entry(fit, fend);
        ngrams.insert(entry.ngram);
        year_min = std::min(year_min, entry.year);
        year_max = std::max(year_max, entry.year);
        var_max = std::max(var_max, entry.var0);
        var_max = std::max(var_max, entry.var1);
    }
    std::cout << "Year: " << year_min << "-" << year_max << std::endl
        << "#ngrams: " << ngrams.size() << std::endl
        << "max(var0, var1): " << var_max << std::endl;


    write_map_file(fname_mapout, ngrams);
}
