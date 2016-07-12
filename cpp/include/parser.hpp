#pragma once

#include <cstdint>

#include <locale>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


using ngram_t           = std::u32string;
using idx_t             = std::uint64_t;
using var_t             = std::uint64_t;
using year_t            = std::uint16_t;
using ngram_idx_map_t   = std::unordered_map<ngram_t, idx_t>;
using idx_ngram_map_t   = std::vector<ngram_t>;
using ngram_ngram_map_t = std::unordered_map<ngram_t, ngram_t>;
using ngram_set_t       = std::set<ngram_t>;
using calc_t            = double;

static_assert(sizeof(double) == 8, "double isn't 64bit :(");

struct entry {
    var_t        var0;
    var_t        var1;
    ngram_t      ngram;
    year_t       year;
    std::uint8_t _padding[6];

    entry(var_t var0_, var_t var1_, const ngram_t& ngram_, year_t year_)
        : var0(var0_),
        var1(var1_),
        ngram(ngram_),
        year(year_) {}

    entry(var_t var0_, var_t var1_, ngram_t&& ngram_, year_t year_)
        : var0(var0_),
        var1(var1_),
        ngram(std::move(ngram_)),
        year(year_) {}
};

entry parse_line_to_entry(const char*& it, const char* end);
std::pair<idx_ngram_map_t, ngram_idx_map_t> parse_map_file(const std::string& fname);
ngram_ngram_map_t parse_trans_file(const std::string& fname);

std::u32string normalize(const std::u32string& s, const std::locale& loc);

void write_map_file(const std::string& fname, const ngram_set_t& ngrams);
