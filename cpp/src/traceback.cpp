#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

#include <boost/locale.hpp>

#include "parser.hpp"
#include "utils.hpp"
#include "tracer.hpp"
#include "wavelet_tree.hpp"

struct tracer_impl_print : tracer_impl_limiter {
    using srs_t = std::unordered_map<superroot_ptr_t, float>;

    const idx_ngram_map_t* idxmap;
    srs_t srs;
    srs_t srs_hist;

    tracer_impl_print(const idx_ngram_map_t* idxmap_, std::size_t depth, std::size_t maxdepth, std::size_t begin, std::size_t end, float minweight)
        : tracer_impl_limiter(depth, maxdepth, begin, end, minweight),
        idxmap(idxmap_) {}

    void down_pre(const tracer_types::next_down_t& current) {
        srs.clear();
        std::cout << "  #nodes=" << current.size() << " => " << std::flush;
    }

    void down_post(const tracer_types::next_down_t&) {
        for (const auto& kv : srs) {
            srs_hist[kv.first] += kv.second;
        }

        if (srs.size() <= 1) {
            std::cout << "." << std::endl;
        } else {
            std::cout << (srs.size() - 1) << ":" << std::endl;
            print_best(srs);
            std::cout << "    --------" << std::endl;
            print_best(srs_hist);
        }
    }

    void found_superroot(const superroot_ptr_t& s, float weight) {
        srs[s] += weight;
    }

    void print_best(srs_t srs_local) {
        for (std::size_t x = 0; x < 20 && !srs_local.empty(); ++x) {
            auto it = std::max_element(srs_local.begin(), srs_local.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.second < rhs.second;
            });
            std::cout
                << "    "
                << boost::locale::conv::utf_to_utf<char>((*idxmap)[it->first->i])
                << ","
                << it->first->i
                << ","
                << it->second
                << std::endl;
            srs_local.erase(it);
        }
    }
};

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_map;
    std::string fname_index;
    std::string query;
    year_t ylength;
    std::size_t maxdepth;
    std::size_t begin;
    std::size_t end;
    float minweight;
    auto desc = po_create_desc();
    desc.add_options()
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("index", po::value(&fname_index)->required(), "index file")
        ("query", po::value(&query)->required(), "query ngram")
        ("maxdepth", po::value(&maxdepth), "maximum search depth (optional)")
        ("begin", po::value(&begin), "search begin (optional)")
        ("end", po::value(&end), "search end (optional)")
        ("minweight", po::value(&minweight), "mininum weight during up-tracing (optional)")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "traceback")) {
        return 1;
    }

    if (!is_power_of_2(ylength)) {
        std::cerr << "ylength has to be a power of 2!" << std::endl;
        return 1;
    }
    std::size_t depth = power_of_2(ylength);
    if (vm.count("maxdepth")) {
        if (maxdepth > depth) {
            std::cerr << "maxdepth should be less or equal to " << depth << std::endl;
            return 1;
        }
    } else {
        maxdepth = depth;
    }
    if (!vm.count("begin")) {
        begin = 0;
    }
    if (!vm.count("end")) {
        end = ylength;
    }
    if (begin >= end) {
        std::cerr << "begin < end!" << std::endl;
        return 1;
    }
    if (end > ylength) {
        std::cerr << "end <= ylength!" << std::endl;
        return 1;
    }
    if (!vm.count("minweight")) {
        minweight = 0.f;
    }
    if (minweight < 0.f) {
        std::cerr << "0.0 <= minweight!" << std::endl;
        return 1;
    }

    idx_ngram_map_t idxmap;
    ngram_idx_map_t ngmap;
    std::tie(idxmap, ngmap) = parse_map_file(fname_map);

    std::cout << "open index file..." << std::endl;
    auto findex = std::make_shared<boost::interprocess::managed_mapped_file>(
        boost::interprocess::open_only,
        fname_index.c_str()
    );
    index_stored_t index(findex, idxmap.size());
    std::cout << "done" << std::endl;

    auto query_utf32 = boost::locale::conv::utf_to_utf<char32_t>(query);
    auto ngram_it = ngmap.find(query_utf32);
    if (ngram_it == ngmap.end()) {
        std::cerr << "unkown ngram" << std::endl;
        return 1;
    }
    std::size_t i = ngram_it->second;

    std::cout << "Run traceback:" << std::endl;
    tracer<tracer_impl_print> t(&index, {&idxmap, depth, maxdepth, begin, end, minweight});
    t(i);
}
