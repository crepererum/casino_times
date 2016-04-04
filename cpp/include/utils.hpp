#pragma once

#include <cstdint>

#include <locale>
#include <string>

#include <boost/program_options.hpp>

// http://stackoverflow.com/a/10791845/1718219
#define XSTR(x) STR(x)
#define STR(x) #x

constexpr bool is_power_of_2(std::size_t v) {
    // see https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    return v && !(v & (v - 1));
}

boost::program_options::options_description po_create_desc();
int po_fill_vm(
    const boost::program_options::options_description& desc,
    boost::program_options::variables_map& vm,
    const int argc,
    char* const* argv,
    const std::string& program_name
);

int gen_locale(std::locale& loc);
