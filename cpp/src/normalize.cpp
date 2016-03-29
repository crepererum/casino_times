#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

using ngram_idx_map_t = std::unordered_map<ngram_t, idx_t>;

int main(int argc, char** argv) {
    std::locale loc;
    if (gen_locale(loc)) {
        return 1;
    }

    std::string fname_out1;
    std::string fname_mapin;
    std::string fname_mapout;
    auto desc = po_create_desc();
    desc.add_options()
        ("mapin", po::value(&fname_mapin)->required(), "ngram map file to read")
        ("mapout", po::value(&fname_mapout)->required(), "ngram map file to write")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "normalize")) {
        return 1;
    }

    ngram_idx_map_t ngmap = parse_map_file(fname_mapin);

    ngram_set_t ngset_norm;
    for (const auto& kv : ngmap) {
        ngset_norm.insert(normalize(kv.first, loc));
    }

    write_map_file(fname_mapout, ngset_norm);

    std::cout << "#Entries(total)=" << ngmap.size() << std::endl
        << "#Entries(written)=" << ngset_norm.size() << std::endl;
}
