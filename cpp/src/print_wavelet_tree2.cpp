#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>

#include "utils.hpp"
#include "wavelet_tree.hpp"

namespace po = boost::program_options;

using counter_t = std::unordered_map<node_ptr_t, std::size_t>;
using nodeset_t = std::unordered_set<node_ptr_t>;
using nodesets_t = std::vector<std::vector<nodeset_t>>;

void walk_down(const node_ptr_t& n, std::size_t l, std::size_t i, nodesets_t& nss, counter_t& counter) {
    if (n != nullptr) {
        nss[l][i].insert(n);
        ++counter[n];
        for (std::size_t d = 0; d < n->children.size(); ++d) {
            walk_down(n->children[d], l + 1, (i * n->children.size()) + d, nss, counter);
        }
    }
}

int main(int argc, char** argv) {
    std::string fname_map;
    std::string fname_tikz;
    std::string fname_index;
    std::vector<std::string> ngrams;
    std::size_t ylength;
    auto desc = po_create_desc();
    desc.add_options()
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("tikzfile", po::value(&fname_tikz)->required(), "tikz file that represents the index")
        ("index", po::value(&fname_index)->required(), "index file")
        ("ngram", po::value(&ngrams)->required()->multitoken(), "ngram to print (can be used multiple times)")
        ("ylength", po::value(&ylength)->required(), "number of years in index")
    ;

    po::variables_map vm;
    if (po_fill_vm(desc, vm, argc, argv, "print_wavelet_tree")) {
        return 1;
    }

    auto idxmap = std::make_shared<idx_ngram_map_t>();
    ngram_idx_map_t ngmap;
    std::tie(*idxmap, ngmap) = parse_map_file(fname_map);

    std::cout << "search ngrams..." << std::endl;
    std::set<std::size_t> indices;
    for (const auto& ng : ngrams) {
        auto ng_utf32 = boost::locale::conv::utf_to_utf<char32_t>(ng);
        auto it = ngmap.find(ng_utf32);
        if (it != ngmap.end()) {
            indices.insert(it->second);
        } else {
            std::cerr << "unable to find ngram: " << ng << std::endl;
            return 1;
        }
    }
    std::cout << "done" << std::endl;

    std::cout << "open index file..." << std::endl;
    auto findex = std::make_shared<boost::interprocess::managed_mapped_file>(
        boost::interprocess::open_only,
        fname_index.c_str()
    );
    index_stored_t index(findex, idxmap->size());
    std::cout << "done" << std::endl;

    const std::size_t depth = power_of_2(ylength);
    std::ofstream out(fname_tikz);

    std::cout << "tracing..." << std::endl;
    std::vector<superroot_ptr_t> srs{};
    counter_t counter{};
    nodesets_t nss(depth);
    for (std::size_t l = 0; l < depth; ++l) {
        nss[l].resize(1ul << l);
    }
    out << "\\begin{scope}[sgroup = " << indices.size() << "]" << std::endl;
    std::size_t sr_idx = 0;
    for (std::size_t i : indices) {
        const auto& s = (*index.superroots)[i];
        srs.emplace_back(s);
        walk_down(s->root, 0, 0, nss, counter);
        out << "    \\mysr{" << boost::locale::conv::utf_to_utf<char>((*idxmap)[i]) << "}{" << sr_idx << "}" << std::endl;
        ++sr_idx;
    }
    out << "\\end{scope}" << std::endl << std::endl;
    std::cout << "done" << std::endl;

    std::cout << "groups..." << std::endl;
    std::unordered_map<node_ptr_t, std::tuple<std::size_t, std::size_t, std::size_t>> npos{};
    for (std::size_t l = 0; l < nss.size(); ++l) {
        const auto& level = nss[l];
        for (std::size_t g = 0; g < level.size(); ++g) {
            const auto& group = level[g];
            std::vector<node_ptr_t> sorted(group.begin(), group.end());
            std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
                return lhs->x < rhs->x;
            });

            float min = sorted[0]->x;
            float max = sorted[sorted.size() - 1]->x;
            float div = max - min;
            if (min == max) {
                div = 1.0f;
                min = min - 0.5f;
            }

            out << "\\begin{scope}[group = " << l << " " << g << " " << group.size() << "]" << std::endl;
            for (std::size_t n = 0; n < sorted.size(); ++n) {
                const auto& node = sorted[n];
                out << "    \\mynode{" << ((node->x - min) / div) << "}{" << n << "}{g" << l << "g" << g << "}" << std::endl;
                npos[node] = std::make_tuple(l, g, n);
            }
            out << "\\end{scope}" << std::endl << std::endl;
        }
    }
    std::cout << "done" << std::endl;

    std::cout << "links..." << std::endl;
    out << "\\begin{scope}[zlevel = llinks]" << std::endl;
    sr_idx = 0;
    for (std::size_t i : indices) {
        const auto& s = (*index.superroots)[i];
        const auto& pos_r = npos[s->root];
        out << "    \\link"
            << "{s;" << sr_idx << ".south}"
            << "{g" << std::get<0>(pos_r) << "g" << std::get<1>(pos_r) << ";" << std::get<2>(pos_r) << ";p}"
            << "{1};" << std::endl;
        ++sr_idx;
    }
    for (const auto& level : nss) {
        for (const auto& group : level) {
            for (const auto& node : group) {
                const auto& pos_node = npos[node];
                if (node->children[0] != nullptr) {
                    const auto& pos_c0 = npos[node->children[0]];
                    out << "    \\link"
                        << "{g" << std::get<0>(pos_node) << "g" << std::get<1>(pos_node) << ";" << std::get<2>(pos_node) << ";c0}"
                        << "{g" << std::get<0>(pos_c0) << "g" << std::get<1>(pos_c0) << ";" << std::get<2>(pos_c0) << ";p}"
                        << "{" << counter[node] << "};" << std::endl;
                }
                if (node->children[1] != nullptr) {
                    const auto& pos_c1 = npos[node->children[1]];
                    out << "    \\link"
                        << "{g" << std::get<0>(pos_node) << "g" << std::get<1>(pos_node) << ";" << std::get<2>(pos_node) << ";c1}"
                        << "{g" << std::get<0>(pos_c1) << "g" << std::get<1>(pos_c1) << ";" << std::get<2>(pos_c1) << ";p}"
                        << "{" << counter[node] << "};" << std::endl;
                }
            }
        }
    }
    out << "\\end{scope}" << std::endl << std::endl;
    std::cout << "done" << std::endl;
}
