#pragma once

#include <cstdint>

#include <locale>
#include <string>
#include <unordered_map>


using ngram_t         = std::u32string;
using idx_t           = std::uint64_t;
using var_t           = std::uint64_t;
using year_t          = std::uint16_t;
using ngram_idx_map_t = std::unordered_map<ngram_t, idx_t>;
using ngram_ngram_map_t = std::unordered_map<ngram_t, ngram_t>;

struct entry {
    var_t        var0;
    var_t        var1;
    ngram_t      ngram;
    year_t       year;
    std::uint8_t _padding[6];
};

entry parse_line_to_entry(const char*& it, const char* end);
ngram_idx_map_t parse_map_file(const std::string& fname);
ngram_ngram_map_t parse_trans_file(const std::string& fname);

std::u32string normalize(const std::u32string& s, const std::locale& loc);
