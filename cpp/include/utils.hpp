#pragma once

#include <cstdint>

constexpr bool is_power_of_2(std::size_t v) {
    // see https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    return v && !(v & (v - 1));
}
