#include <cstdint>

#include <iterator>
#include <string>
#include <utility>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/locale.hpp>

#include "parser.hpp"


std::pair<const char*, const char*> get_line(const char*& it, const char* end) {
    // skip empty lines
    while (it != end && (*it == '\n' || *it == '\r')) {
        ++it;
    }

    auto lbegin = it;

    // fetch until line or file ends
    while (it != end && *it != '\n' && *it != '\r') {
        ++it;
    }

    auto lend = it;

    // skip possible endline character
    if (it != end) {
        ++it;
    }

    // return converted string
    return std::make_pair(lbegin, lend);
}

template <typename I>
std::string get_column(I& it, const I end) {
    std::string result;
    while (it != end && *it != '\t') {
        result.push_back(*it);
        ++it;
    }

    if (it != end) {
        ++it;
    }

    return result;
}

entry parse_entry(std::pair<const char*, const char*> line){
    auto it = std::get<0>(line);
    auto end = std::get<1>(line);
    auto s0 = get_column(it, end);
    auto s1 = get_column(it, end);
    auto s2 = get_column(it, end);
    auto s3 = get_column(it, end);
    return {
        .ngram=boost::locale::conv::utf_to_utf<char32_t>(s0),
        .year=static_cast<std::uint16_t>(std::stoul(s1)),
        .var0=std::stoull(s2),
        .var1=std::stoull(s3),
    };
}

entry parse_line_to_entry(const char*& it, const char* end) {
    return parse_entry(get_line(it, end));
}

ngram_idx_map_t parse_map_file(const std::string& fname) {
    boost::iostreams::mapped_file input(fname, boost::iostreams::mapped_file::mapmode::readonly);
    if (!input.is_open()) {
        // XXX: exception!
        return {};
    }

    ngram_idx_map_t result;
    auto fit = input.const_data();
    auto fend = fit + input.size();
    idx_t idx = 0;
    while (fit != fend) {
        auto line = get_line(fit, fend);
        std::string s8(std::get<0>(line), std::get<1>(line));

        if (!s8.empty()) {
            auto ngram = boost::locale::conv::utf_to_utf<char32_t>(s8);
            if (result.find(ngram) != result.end()) {
                // XXX: exception
            }
            result.insert(std::make_pair(std::move(ngram), idx));
            ++idx;
        }
    }

    return result;
}

ngram_ngram_map_t parse_trans_file(const std::string& fname) {
    std::string sep(" -> ");
    boost::iostreams::mapped_file input(fname, boost::iostreams::mapped_file::mapmode::readonly);
    if (!input.is_open()) {
        // XXX: exception!
        return {};
    }

    ngram_ngram_map_t result;
    auto fit = input.const_data();
    auto fend = fit + input.size();
    while (fit != fend) {
        auto line = get_line(fit, fend);
        std::string s8(std::get<0>(line), std::get<1>(line));

        if (!s8.empty()) {
            auto pos_sep = s8.find(sep);
            if (pos_sep != std::string::npos) {
                auto ngram0 = boost::locale::conv::utf_to_utf<char32_t>(s8.substr(0, pos_sep));
                auto ngram1 = boost::locale::conv::utf_to_utf<char32_t>(s8.substr(pos_sep + sep.length()));
                result.insert(std::make_pair(std::move(ngram0), std::move(ngram1)));
            }
        }
    }

    return result;
}

std::u32string normalize(const std::u32string& s, const std::locale& loc) {
    auto utf8in = boost::locale::conv::utf_to_utf<char>(s);
    auto nfkc = boost::locale::normalize(utf8in, boost::locale::norm_nfkc, loc);
    auto lowered = boost::locale::to_lower(nfkc, loc);
    auto utf32out = boost::locale::conv::utf_to_utf<char32_t>(lowered);
    return utf32out;
}
