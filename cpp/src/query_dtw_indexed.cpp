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
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "dtw.hpp"
#include "dtw_index.hpp"
#include "parser.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

struct queue_element_t {
    calc_t dist;
    std::size_t idx;

    queue_element_t(calc_t dist_, std::size_t idx_) : dist(dist_), idx(idx_) {}
};

struct cmp_queue {
    bool operator()(const queue_element_t& a, const queue_element_t& b) {
        return a.dist > b.dist;
    }
};

using queue_t = std::priority_queue<queue_element_t, std::vector<queue_element_t>, cmp_queue>;


struct result_entry_t {
    std::size_t idx;
    calc_t dist;

    result_entry_t(std::size_t idx_, calc_t dist_) : idx(idx_), dist(dist_) {}
    result_entry_t(const queue_element_t& obj) : idx(obj.idx), dist(obj.dist) {}
};

struct result_t {
    std::vector<result_entry_t> entries;
    std::size_t counter_dtw;
    std::size_t counter_fetched;

    result_t() : entries(), counter_dtw(0), counter_fetched(0) {}
};

result_t run_query(const calc_t* base, year_t ylength, std::size_t i, std::size_t r, std::size_t limit, std::size_t n, const tree_t* tree, const downstorage_t* down) {
    auto q_lu = get_lu(base, ylength, i, r);
    box_t q_box;
    get_downsampled_l::f(q_lu.first.data(), ylength, q_box.min_corner());
    get_downsampled_u::f(q_lu.second.data(), ylength, q_box.max_corner());

    calc_t normfactor = static_cast<calc_t>(ylength >> dtw_index_resolution_shift);
    box_t s_box;
    std::size_t usable_limit = std::min(limit, n);
    dtw_simple mydtw_simple(base, ylength, r);
    queue_t queue;
    queue_t temp;
    result_t result;
    std::size_t s_j;
    for (tree_t::const_query_iterator it = tree->qbegin(boost::geometry::index::nearest(q_box, static_cast<unsigned int>(n))); it != tree->qend(); ++it) {
        std::tie(s_box, s_j) = *it;
        auto mindist = normfactor * mindist_unnorm(q_box, s_box);

        while (!queue.empty() && queue.top().dist < mindist) {
            auto p = queue.top();
            queue.pop();

            while (!temp.empty() && temp.top().dist < p.dist) {
                result.entries.emplace_back(std::move(temp.top()));
                temp.pop();
                if (result.entries.size() >= usable_limit) {
                    return result;
                }
            }

            temp.push(queue_element_t(mydtw_simple.calc(i, p.idx), p.idx));
            ++result.counter_dtw;
        }

        queue.emplace(normfactor * lb_paa_unnorm(q_box, (*down)[s_j]), s_j);
        ++result.counter_fetched;
    }
    while (!queue.empty()) {
        auto p = queue.top(); queue.pop();

        while (!temp.empty() && temp.top().dist < p.dist) {
            result.entries.emplace_back(std::move(temp.top()));
            temp.pop();
            if (result.entries.size() >= usable_limit) {
                return result;
            }
        }

        temp.push(queue_element_t(mydtw_simple.calc(i, p.idx), p.idx));
        ++result.counter_dtw;
    }
    while (!temp.empty()) {
        result.entries.emplace_back(std::move(temp.top()));
        temp.pop();
        if (result.entries.size() >= usable_limit) {
            return result;
        }
    }

    return result;
}


int main(int argc, char** argv) {
    std::string fname_binary;
    std::string fname_map;
    std::string fname_index;
    std::string query;
    std::size_t r;
    std::size_t limit;
    year_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("binary", po::value(&fname_binary)->required(), "input binary file for var")
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("ylength", po::value(&ylength)->required(), "number of years to store")
        ("r", po::value(&r)->required(), "radius of Sakoe-Chiba Band")
        ("limit", po::value(&limit)->required(), "number of ngrams to look for")
        ("query", po::value(&query)->required(), "query ngram")
        ("index", po::value(&fname_index)->required(), "index file")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "query_dtw_indexed")) {
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

    auto query_utf32 = boost::locale::conv::utf_to_utf<char32_t>(query);
    auto ngram_it = ngmap.find(query_utf32);
    if (ngram_it == ngmap.end()) {
        std::cerr << "unkown ngram" << std::endl;
        return 1;
    }
    std::size_t i = ngram_it->second;

    boost::iostreams::mapped_file_params params;
    params.path   = fname_binary;
    params.flags  = boost::iostreams::mapped_file::mapmode::readonly;
    params.length = static_cast<std::size_t>(n * ylength * sizeof(calc_t));
    params.offset = 0;
    boost::iostreams::mapped_file input(params);
    if (!input.is_open()) {
        std::cerr << "cannot read input file" << std::endl;
        return 1;
    }
    auto base = reinterpret_cast<const calc_t*>(input.const_data());

    std::cout << "open index file..." << std::endl;
    std::size_t find_count;
    downstorage_t* down;
    tree_t* tree;
    boost::interprocess::managed_mapped_file findex(boost::interprocess::open_only, fname_index.c_str());
    allocator_point_t alloc_point(findex.get_segment_manager());
    allocator_node_t alloc_node(findex.get_segment_manager());
    std::tie(down, find_count) = findex.find<downstorage_t>("down");
    std::tie(tree, find_count) = findex.find<tree_t>("rtree");
    if (down == nullptr || tree == nullptr) {
        std::cerr << "cannot find index entries!" << std::endl;
        return 1;
    }
    std::cout << "done" << std::endl;

    auto result = run_query(base, ylength, i, r, limit, n, tree, down);

    std::cout << "stats:" << std::endl
        << "  n        = " << n << std::endl
        << "  #fetched = " << result.counter_fetched << std::endl
        << "  #dtw     = " << result.counter_dtw << std::endl
        << std::endl;

    constexpr std::size_t colw0 = 10;
    constexpr std::size_t colw1 = 10;
    constexpr std::size_t colw2 = 10;
    std::cout
        << "| "
        << std::setw(colw0) << "ngram"
        << " | "
        << std::setw(colw1) << "id"
        << " | "
        << std::setw(colw2) << "distance"
        << " |"
        << std::endl;
    std::cout
        << "|-"
        << std::string(colw0, '-')
        << "-|-"
        << std::string(colw1, '-')
        << "-|-"
        << std::string(colw2, '-')
        << "-|"
        << std::endl;
    for (const auto& element : result.entries) {
        std::cout
            << "| "
            << std::setw(colw0) << boost::locale::conv::utf_to_utf<char>(idxmap[element.idx])
            << " | "
            << std::setw(colw1) << element.idx
            << " | "
            << std::setw(colw2) << element.dist
            << " |"
            << std::endl;
    }
}
