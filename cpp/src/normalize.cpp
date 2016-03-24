#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

using ngram_idx_map_t = std::unordered_map<ngram_t, idx_t>;

int main(int argc, char** argv) {
    // before we start, check if we're working on an UTF8 system
    boost::locale::generator gen;
    std::locale loc = gen("");
    if (!std::use_facet<boost::locale::info>(loc).utf8()) {
        std::cerr << "sorry, this program only works on UTF8 systems" << std::endl;
    }

    std::string fname_out1;
    std::string fname_mapin;
    std::string fname_mapout;
    po::options_description desc("all the options");
    desc.add_options()
        ("mapin", po::value(&fname_mapin)->required(), "ngram map file to read")
        ("mapout", po::value(&fname_mapout)->required(), "ngram map file to write")
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

    ngram_idx_map_t ngmap = parse_map_file(fname_mapin);

    std::set<ngram_t> ngset_norm;
    for (const auto& kv : ngmap) {
        ngset_norm.insert(normalize(kv.first, loc));
    }

    std::ofstream mapfile(fname_mapout);
    for (const auto& ngram : ngset_norm) { // they are sorted
        mapfile << boost::locale::conv::utf_to_utf<char>(ngram) << '\n';
    }

    std::cout << "#Entries(total)=" << ngmap.size() << std::endl
        << "#Entries(written)=" << ngset_norm.size() << std::endl;
}
