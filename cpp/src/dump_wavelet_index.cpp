#include <string>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/program_options.hpp>

#include "parser.hpp"
#include "utils.hpp"
#include "wavelet_tree.hpp"
#include "wavelet_transformer.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_index;
    std::string fname_map;
    std::string fname_out;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("index", po::value(&fname_index)->required(), "index file")
        ("binary", po::value(&fname_out)->required(), "output binary file")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "dump_wavelet_index")) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }

    idx_ngram_map_t idxmap;
    ngram_idx_map_t ngmap;
    std::tie(idxmap, ngmap) = parse_map_file(fname_map);
    std::size_t n = ngmap.size();

    std::cout << "open index file..." << std::endl;
    std::size_t find_count;
    superroot_vector_t* superroots;
    auto findex = std::make_shared<boost::interprocess::managed_mapped_file>(
        boost::interprocess::open_only,
        fname_index.c_str()
    );
    auto segment_manager = findex->get_segment_manager();
    allocator_superroot_ptr_t allocator_superroot(segment_manager);
    std::tie(superroots, find_count) = findex->find<superroot_vector_t>("superroots");
    if (superroots == nullptr) {
        std::cerr << "cannot find index data!" << std::endl;
        return 1;
    }
    std::cout << "done" << std::endl;

    std::cout << "open output file..." << std::endl;
    auto output = open_raw_file(fname_out, n * ylength * sizeof(calc_t), true, true);
    if (!output.is_open()) {
        std::cerr << "cannot write output file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<calc_t*>(output.data());
    std::cout << "done" << std::endl;

    std::cout << "dump data..." << std::endl;
    std::size_t depth = power_of_2(ylength);
    transformer trans(ylength, depth);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t offset = i * ylength;
        trans.superroot = (*superroots)[i];
        trans.tree_to_data(base + offset);
    }
    std::cout << "done" << std::endl;
}
