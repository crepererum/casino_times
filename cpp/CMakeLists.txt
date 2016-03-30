# generic CMake setup
cmake_minimum_required (VERSION 3.0)
include (ExternalProject)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project metadata
project (ngrams)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -gsplit-dwarf -pthread -Wall -Wextra -Wconversion -Wsign-conversion -Wpedantic -Wno-unknown-pragmas")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# detect clang
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-c99-extensions -Wno-global-constructors -Wno-missing-prototypes -fcolor-diagnostics -fdiagnostics-show-category=name")
endif ()

# faster linker
execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
if ("${LD_VERSION}" MATCHES "GNU gold")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
else ()
    message(WARNING "GNU gold linker isn't available, using the default system linker.")
endif ()

# includes
include_directories (
    "include"
)

# lib: parser
add_library (parser STATIC "src/parser.cpp")
target_link_libraries (
    parser
    boost_locale
)

# lib: utils
add_library (utils STATIC "src/utils.cpp")
target_link_libraries (
    utils
    boost_locale
)

# executable: create
add_executable (create "src/create.cpp")
target_link_libraries (
    create
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)

# executable: normalize
add_executable (normalize "src/normalize.cpp")
target_link_libraries (
    normalize
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)

# executable: query_dtw
add_executable (query_dtw "src/query_dtw.cpp")
target_link_libraries (
    query_dtw
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)

# executable: scan
add_executable (scan "src/scan.cpp")
target_link_libraries (
    scan
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)

# executable: store
add_executable (store "src/store.cpp")
target_link_libraries (
    store
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)
