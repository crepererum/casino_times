# generic CMake setup
cmake_minimum_required (VERSION 3.0)
include (ExternalProject)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project metadata
project (ngrams)

# bunch of external projects
externalproject_add (
    project_libsimdpp
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libsimdpp"
    GIT_REPOSITORY "https://github.com/p12tic/libsimdpp.git"
    GIT_TAG "v2.0-rc2"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/simdpp <INSTALL_DIR>/simdpp
    TEST_COMMAND ""
)

externalproject_add (
    project_wavelib
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/wavelib"
    GIT_REPOSITORY "https://github.com/rafat/wavelib.git"
    GIT_TAG "a2c709715b5987247bf0182edc01d9791ce33ae2"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DBUILD_UT=Off
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/Bin <INSTALL_DIR>/lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/header <INSTALL_DIR>/include
    TEST_COMMAND ""
)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx2 -mfma -g -fno-omit-frame-pointer -fsized-deallocation -pthread -Wall -Wextra -Wconversion -Wsign-conversion -Wpedantic -Wno-unknown-pragmas")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address")
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

include_directories (
    SYSTEM
    "${CMAKE_CURRENT_BINARY_DIR}/libsimdpp"
    "${CMAKE_CURRENT_BINARY_DIR}/wavelib/include"
)

# lib paths
link_directories (
    "${CMAKE_CURRENT_BINARY_DIR}/wavelib/lib"
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
    boost_program_options
)

# executable: calc_dtw_simple
add_executable (calc_dtw_simple "src/calc_dtw_simple.cpp")
target_link_libraries (
    calc_dtw_simple
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
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

# executable: dump_wavelet_index
add_executable (dump_wavelet_index "src/dump_wavelet_index.cpp")
target_link_libraries (
    dump_wavelet_index
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
    wavelib
)
add_dependencies (dump_wavelet_index project_wavelib)

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

# executable: index_dtw
add_executable (index_dtw "src/index_dtw.cpp")
target_link_libraries (
    index_dtw
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)
add_dependencies (index_dtw project_libsimdpp)

# executable: index_wavelet
add_executable (index_wavelet "src/index_wavelet.cpp")
target_link_libraries (
    index_wavelet
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
    wavelib
)
add_dependencies (index_wavelet project_wavelib)

# executable: print_wavelet_tree
add_executable (print_wavelet_tree "src/print_wavelet_tree.cpp")
target_link_libraries (
    print_wavelet_tree
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)

# executable: query_dtw_indexed
add_executable (query_dtw_indexed "src/query_dtw_indexed.cpp")
target_link_libraries (
    query_dtw_indexed
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)
add_dependencies (query_dtw_indexed project_libsimdpp)

# executable: query_dtw_simple
add_executable (query_dtw_simple "src/query_dtw_simple.cpp")
target_link_libraries (
    query_dtw_simple
    parser
    utils
    boost_iostreams
    boost_locale
    boost_program_options
)
add_dependencies (query_dtw_simple project_libsimdpp)

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
