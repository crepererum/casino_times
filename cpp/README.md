# C++ components of the nGram project

## Requirements

- C++14 compiler and stdlib
- Boost (tested with 1.60)
- Linux
- CMake
- Make (or Ninja)

## Building

    mkdir build
    cd build
    cmake ..
    make

You can use Ninja as a build tool instead.

## Usage
The produced binaries are static ones, so you can use them on every Linux system which supports the emitted instruction
set:

    cd build
    ./whatever_tool --help


## Troubleshooting

### Illegal instruction
That might happen because we force the compiler to use certain instruction sets (e.g. AVX2). To fix that problem,
change the compiler flags in `CmakeLists.txt` and change the SIMDPP settings in `include/dtw.hpp`

### SEGFAULT
Check if the nGram map file and binary data file should go together.

### Weird results
Does the tool use `uint64` or `float64` data? For example the `store` tool emits `uint64` while all DTW tools require
`float64` data. So you need to convert the data first, e.g. by using one of the Julia scripts which are also part of
this repository.
