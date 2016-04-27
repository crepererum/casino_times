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

struct queue_helper_element_t {
    calc_t dist;
    std::size_t idx;

    queue_helper_element_t(calc_t dist_, std::size_t idx_) : dist(dist_), idx(idx_) {}
};

struct cmp_queue_helper {
    bool operator()(const queue_helper_element_t& a, const queue_helper_element_t& b) {
        return a.dist > b.dist;
    }
};

using queue_helper_t = std::priority_queue<queue_helper_element_t, std::vector<queue_helper_element_t>, cmp_queue_helper>;


class alignas(128) queue_t {
    public:
        static constexpr std::size_t max_buffer_size = 1024 * 16;

        queue_t(box_t q_box, const downstorage_t* down, calc_t normfactor): _q_box(q_box), _down(down), _normfactor(normfactor) {
            _buffer.reserve(max_buffer_size);
        }

        void add_to_buffer(std::size_t idx) {
            _buffer.emplace_back(idx);

            if (_buffer.size() >= max_buffer_size) {
                flush_to_queue();
            }
        }

        void flush_to_queue() {
            std::size_t n      = _buffer.size();
            std::size_t n_over = n % width;
            std::size_t n_good = n - n_over;

            for (std::size_t b = 0; b < n_good; b += width) {
                lb_paa_vectorized(b);
            }

            for (std::size_t b = n_good; b < n; ++b) {
                std::size_t idx = _buffer[b];
                _queue.emplace(_normfactor * lb_paa_unnorm(_q_box, (*_down)[idx]), idx);
            }

            _buffer.clear();
        }

        bool has_data() const {
            return !_queue.empty();
        }

        const queue_helper_element_t& top() const {
            return _queue.top();
        }

        queue_helper_element_t pop() {
            queue_helper_element_t top = _queue.top();
            _queue.pop();
            return top;
        }

    private:
        box_t                                               _q_box;
        const downstorage_t*                                _down;
        calc_t                                              _normfactor;
        std::vector<std::size_t>                            _buffer;
        queue_helper_t                                      _queue;

        static constexpr std::size_t width = 16;

        using vector_t  = simdpp::float64<width>;
        using mask_t    = simdpp::mask_float64<width>;
        using indices_t = std::array<std::size_t, width>;
        using bases_t   = std::array<const calc_t*, width>;
        using tmp_t     = std::array<calc_t, width>;

        indices_t          _indices;
        bases_t            _bases;
        alignas(128) tmp_t _tmp;

        void lb_paa_vectorized(std::size_t p) {
            vector_t zero       = simdpp::make_float(0.0);
            vector_t sum        = simdpp::make_float(0.0);
            vector_t normfactor = simdpp::make_float(_normfactor);

            fetch_indices(p);
            fetch_bases();

            for (std::size_t i = 0; i < dtw_index_resolution; ++i) {
                transfer_bases_to_tmp(i);
                vector_t x = simdpp::load(_tmp.data());
                vector_t u = simdpp::make_float(_q_box.max_corner()[i]);
                vector_t l = simdpp::make_float(_q_box.min_corner()[i]);

                mask_t cond_u = simdpp::cmp_gt(x, u);
                mask_t cond_l = simdpp::cmp_lt(x, l);

                vector_t delta_u = simdpp::sub(x, u);
                vector_t delta_l = simdpp::sub(l, x);

                vector_t delta = simdpp::blend(delta_u, zero, cond_u);
                delta = simdpp::blend(delta_l, delta, cond_l);

                sum = simdpp::fmadd(delta, delta, sum);
            }

            sum = simdpp::mul(sum, normfactor);

            simdpp::store(_tmp.data(), sum);
            for (std::size_t i = 0; i < width; ++i) {
                _queue.emplace(_tmp[i], _indices[i]);
            }
        }

        void fetch_indices(std::size_t p) {
            for (std::size_t pd = 0; pd < width; ++pd) {
                _indices[pd] = _buffer[p + pd];
            }
        }

        void fetch_bases() {
            for (std::size_t i = 0; i < width; ++i) {
                _bases[i] = (*_down)[_indices[i]].data();
            }
        }

        void transfer_bases_to_tmp(std::size_t d) {
            for (std::size_t i = 0; i < width; ++i) {
                _tmp[i] = _bases[i][d];
            }
        }
};


class temp_t {
    public:
        static constexpr std::size_t max_buffer_size = 128;

        temp_t(const calc_t* base, year_t ylength, std::size_t r, std::size_t i): _mydtw_simple(base, ylength, r), _mydtw_vectorized(base, ylength, r), _i(i) {
            _buffer.reserve(max_buffer_size);
        }

        void add_to_buffer(std::size_t idx) {
            _buffer.emplace_back(idx);

            if (_buffer.size() >= max_buffer_size) {
                flush_to_queue();
            }
        }

        void flush_to_queue() {
            std::size_t n      = _buffer.size();
            std::size_t n_over = n % static_cast<std::size_t>(dtw_vectorized_shuffled::n);
            std::size_t n_good = n - n_over;

            for (std::size_t b = 0; b < n_good; b += dtw_vectorized_linear::n) {
                for (std::size_t bd = 0; bd < dtw_vectorized_shuffled::n; ++bd) {
                    _vindices[bd] = _buffer[b + bd];
                }

                _dist = _mydtw_vectorized.calc(_i, _vindices);

                simdpp::store(&_vresults, _dist);

                for (std::size_t bd = 0; bd < dtw_vectorized_shuffled::n; ++bd) {
                    _queue.emplace(_vresults[bd], _buffer[b + bd]);
                }
            }

            for (std::size_t b = n_good; b < n; ++b) {
                std::size_t idx = _buffer[b];
                _queue.emplace(_mydtw_simple.calc(_i, idx), idx);
            }

            _buffer.clear();
        }

        bool has_data() const {
            return !_queue.empty();
        }

        const queue_helper_element_t& top() const {
            return _queue.top();
        }

        queue_helper_element_t pop() {
            queue_helper_element_t top = _queue.top();
            _queue.pop();
            return top;
        }

    private:
        dtw_simple                                          _mydtw_simple;
        dtw_vectorized_shuffled                             _mydtw_vectorized;
        std::array<std::size_t, dtw_vectorized_shuffled::n> _vindices;
        dtw_vectorized_shuffled::dist_t                     _dist;
        std::array<calc_t, dtw_vectorized_shuffled::n>      _vresults;
        std::size_t                                         _i;
        std::vector<std::size_t>                            _buffer;
        queue_helper_t                                      _queue;
};


struct result_entry_t {
    std::size_t idx;
    calc_t dist;

    result_entry_t(std::size_t idx_, calc_t dist_) : idx(idx_), dist(dist_) {}
    result_entry_t(const queue_helper_element_t& obj) : idx(obj.idx), dist(std::sqrt(obj.dist)) {}
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
    std::size_t s_j;
    std::size_t usable_limit = std::min(limit, n);
    queue_t queue(q_box, down, normfactor);
    temp_t temp(base, ylength, r, i);
    result_t result;
    for (tree_t::const_query_iterator it = tree->qbegin(boost::geometry::index::nearest(q_box, static_cast<unsigned int>(n))); it != tree->qend(); ++it) {
        std::tie(s_box, s_j) = *it;
        auto mindist = normfactor * mindist_unnorm(q_box, s_box);

        while (queue.has_data() && queue.top().dist < mindist) {
            auto p = queue.top();
            queue.pop();

            while (temp.has_data() && temp.top().dist < p.dist) {
                result.entries.emplace_back(temp.pop());
                if (result.entries.size() >= usable_limit) {
                    return result;
                }
            }

            temp.add_to_buffer(p.idx);
            ++result.counter_dtw;
        }

        queue.add_to_buffer(s_j);
        ++result.counter_fetched;
    }
    queue.flush_to_queue();
    while (queue.has_data()) {
        auto p = queue.top(); queue.pop();

        while (temp.has_data() && temp.top().dist < p.dist) {
            result.entries.emplace_back(temp.pop());
            if (result.entries.size() >= usable_limit) {
                return result;
            }
        }

        temp.add_to_buffer(p.idx);
        ++result.counter_dtw;
    }
    temp.flush_to_queue();
    while (temp.has_data()) {
        result.entries.emplace_back(temp.pop());
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

    auto input = open_raw_file(fname_binary, n * ylength * sizeof(calc_t), false, false);
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
