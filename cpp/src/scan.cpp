#include <cstdint>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    // before we start, check if we're working on an UTF8 system
    boost::locale::generator gen;
    std::locale loc = gen("");
    if (!std::use_facet<boost::locale::info>(loc).utf8()) {
        std::cerr << "sorry, this program only works on UTF8 systems" << std::endl;
    }

    std::string fname_in;
    std::string fname_mapout;
    po::options_description desc("all the options");
    desc.add_options()
        ("file", po::value(&fname_in)->required(), "input ngram file")
        ("map", po::value(&fname_mapout)->required(), "ngram map file to write")
        ("help", "print help message")
    ;

    po::variables_map vm;
    try {
        po::store(
            po::command_line_parser(argc, argv).options(desc).run(),
            vm
        );
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    try {
        po::notify(vm);
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    boost::iostreams::mapped_file input(fname_in, boost::iostreams::mapped_file::mapmode::readonly);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto fit = input.const_data();
    auto fend = fit + input.size();

    std::set<ngram_t> ngrams;
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


    std::ofstream mapfile(fname_mapout);
    for (const auto& ngram : ngrams) { // they are sorted
        mapfile << boost::locale::conv::utf_to_utf<char>(ngram) << '\n';
    }
}
