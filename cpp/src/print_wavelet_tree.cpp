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

class printer {
    public:
        printer(superroot_vector_t* superroots, const std::shared_ptr<idx_ngram_map_t>& idxmap) :
            _superroots(superroots),
            _idxmap(idxmap) {}

        std::ostream& print_begin(std::ostream &out) {
            out << "digraph treegraph {" << std::endl;
            out << "  nodesep=0.1;" << std::endl;
            out << "  ranksep=5;" << std::endl;
            out << "  size=\"25,25\";" << std::endl;
            out << "  splines=false;" << std::endl;
            out << std::endl;

            return out;
        }

        std::ostream& print_end(std::ostream &out) {
            out << "}" << std::endl;

            return out;
        }

        std::ostream& print_tree(std::ostream &out, std::size_t i) {
            superroot_ptr_t superroot = (*_superroots)[i];
            print_superroot(out, superroot, (*_idxmap)[i]);

            std::vector<std::pair<node_ptr_t, std::pair<std::size_t, std::size_t>>> todo;
            todo.push_back(std::make_pair(superroot->root, std::make_pair(0, 0)));

            while (!todo.empty()) {
                auto current = todo.back();
                todo.pop_back();

                print_node(out, current.first, current.second.first, current.second.second);

                if (current.first->child_l) {
                    todo.push_back(std::make_pair(
                        current.first->child_l,
                        std::make_pair(current.second.first + 1, current.second.second << 1)
                    ));
                }
                if (current.first->child_r) {
                    todo.push_back(std::make_pair(
                        current.first->child_r,
                        std::make_pair(current.second.first + 1, (current.second.second << 1) + 1)
                    ));
                }
            }

            return out;
        }

        std::ostream& print_groups(std::ostream &out) {
            for (const auto& kv : _groups) {
                out << "  subgraph cluster_" << kv.first.first << "_" << kv.first.second << " {" << std::endl;
                for (const auto& n : kv.second) {
                    out << "    ";
                    print_address(out, n);
                    out << ";" << std::endl;
                }
                out << "    graph[style=dotted];" << std::endl;
                out << "  }" << std::endl;
                out << std::endl;
            }

            return out;
        }

    private:
        superroot_vector_t*                                                    _superroots;
        std::shared_ptr<idx_ngram_map_t>                                       _idxmap;
        std::unordered_set<node_ptr_t>                                         _printed_nodes;
        std::map<std::pair<std::size_t, std::size_t>, std::vector<node_ptr_t>> _groups;

        template <typename T>
        std::ostream& print_address(std::ostream &out, T* addr) {
            out << "addr" << addr;
            return out;
        }

        template <typename T>
        std::ostream& print_address(std::ostream &out, boost::interprocess::offset_ptr<T> addr) {
            return print_address(out, addr.get());
        }

        std::ostream& print_superroot(std::ostream &out, superroot_ptr_t superroot, const ngram_t& ngram) {
            out << "  ";
            print_address(out, superroot);
            out << " [label=\"" << boost::locale::conv::utf_to_utf<char>(ngram) << "\", shape=circle, fixedsize=true, width=6, height=6, fontsize=80];" << std::endl;

            out << "  ";
            print_address(out, superroot);
            out << ":s -> ";
            print_address(out, superroot->root);
            out << ":t:n;" << std::endl;

            out << std::endl;

            return out;
        }

        std::ostream& print_node(std::ostream &out, node_ptr_t node, std::size_t l, std::size_t idx) {
            if (_printed_nodes.find(node) == _printed_nodes.end()) {
                out << "  ";
                print_address(out, node);
                out << " [shape=none, margin=0, fixedsize=true, width=1.0, height=0.6, fontsize=8, label=<" << std::endl;
                out << "    <table border=\"0\" cellborder=\"1\" cellspacing=\"0\" cellpadding=\"4\">" << std::endl;
                out << "    <tr><td port=\"t\" colspan=\"2\" fixedsize=\"true\" width=\"60\" height=\"20\">" << node->x << "</td></tr>" << std::endl;
                out << "    <tr><td port=\"0\">0</td><td port=\"1\">1</td></tr>" << std::endl;
                out << "  </table>>];" << std::endl;

                if (node->child_l) {
                    out << "  ";
                    print_address(out, node);
                    out << ":0:s -> ";
                    print_address(out, node->child_l);
                    out << ":t:n;" << std::endl;
                }
                if (node->child_r) {
                    out << "  ";
                    print_address(out, node);
                    out << ":1:s -> ";
                    print_address(out, node->child_r);
                    out << ":t:n;" << std::endl;
                }

                out << std::endl;

                _printed_nodes.insert(node);
                _groups[std::make_pair(l, idx)].push_back(node);
            }

            return out;
        }
};

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string fname_map;
    std::string fname_dot;
    std::string fname_index;
    std::vector<std::string> ngrams;
    auto desc = po_create_desc();
    desc.add_options()
        ("map", po::value(&fname_map)->required(), "ngram map file to read")
        ("dotfile", po::value(&fname_dot)->required(), "dot file that represents the index")
        ("index", po::value(&fname_index)->required(), "index file")
        ("ngram", po::value(&ngrams)->required()->multitoken(), "ngram to print (can be used multiple times)")
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

    std::cout << "emit dot file..." << std::endl;
    std::ofstream out(fname_dot);
    printer pr(superroots, idxmap);
    pr.print_begin(out);
    for (std::size_t i : indices) {
        pr.print_tree(out, i);
    }
    pr.print_groups(out);
    pr.print_end(out);
    std::cout << "done" << std::endl;
}
