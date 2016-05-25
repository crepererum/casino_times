# generic CMake setup
cmake_minimum_required (VERSION 3.0)
include (ExternalProject)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project metadata
project (ngrams)

# bunch of external projects
set (external_flags "-fPIC -fdata-sections -ffunction-sections")
externalproject_add (
    project_libicu
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libicu"
    URL "http://download.icu-project.org/files/icu4c/57.1/icu4c-57_1-src.tgz"
    URL_HASH SHA512=a3c701e9c81622db545bcf93f315c7b13159750f43f009d0aec59ceae3a8e1ccb751826d4b8a7387aca47f38bff2a85816b1a123b07d2bf731558c7b66e47b8a
    CONFIGURE_COMMAND cd <SOURCE_DIR>/source && env "CFLAGS=${external_flags}" "CXXFLAGS=${external_flags}" ./runConfigureICU Linux --enable-static --disable-shared --disable-dyload
    BUILD_COMMAND cd <SOURCE_DIR>/source && make
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/source/lib <INSTALL_DIR>/lib
    TEST_COMMAND ""
)

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
    project_libtcmalloc
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libtcmalloc"
    URL "https://github.com/gperftools/gperftools/releases/download/gperftools-2.5/gperftools-2.5.tar.gz"
    URL_HASH SHA512=3e80c0d8816425d65673cf5cff28c6801b0c675501263f830828600f37ac01199fd775c2c51c9ccfa53fd4f9eadc09f7b2c73b31d57161e2cdfee2bd1868afa7
    CONFIGURE_COMMAND cd <SOURCE_DIR> && env "CFLAGS=${external_flags}" "CXXFLAGS=${external_flags}" ./configure --disable-cpu-profiler --disable-heap-profiler --disable-heap-checker --enable-dynamic-sized-delete-support --enable-sized-delete --enable-static --disable-shared --prefix=<INSTALL_DIR>
    BUILD_COMMAND cd <SOURCE_DIR> && make
    INSTALL_COMMAND cd <SOURCE_DIR> && make install
    TEST_COMMAND ""
)

externalproject_add (
    project_wavelib
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/wavelib"
    GIT_REPOSITORY "https://github.com/rafat/wavelib.git"
    GIT_TAG "a2c709715b5987247bf0182edc01d9791ce33ae2"
    CMAKE_ARGS "-DCMAKE_C_FLAGS=${external_flags}" "-DCMAKE_CXX_FLAGS=${external_flags}" -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DBUILD_UT=Off
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/Bin <INSTALL_DIR>/lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/header <INSTALL_DIR>/include
    TEST_COMMAND ""
)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx2 -mfma -ffast-math -fPIC -fdata-sections -ffunction-sections -g -fno-omit-frame-pointer -fsized-deallocation -pthread -Wall -Wextra -Wconversion -Wsign-conversion -Wpedantic -Wno-unknown-pragmas")
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

# we want static executables
set (CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set (BUILD_SHARED_LIBRARIES OFF)
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--gc-sections -Wl,--no-export-dynamic")  # no idea why the dynamic export stuff is required

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
    "${CMAKE_CURRENT_BINARY_DIR}/libicu/lib"
    "${CMAKE_CURRENT_BINARY_DIR}/libtcmalloc/lib"
    "${CMAKE_CURRENT_BINARY_DIR}/wavelib/lib"
)

# list of all required libs
# INFO: We don't differ between the requirements of the different parts.
#       That makes the management of this build scripts way easier.
set (
    ngrams_dep
    project_libicu
    project_libsimdpp
    project_libtcmalloc
    project_wavelib
)
set (
    ngrams_libs
    boost_iostreams
    boost_locale
    boost_program_options
    boost_system
    boost_thread
    icui18n
    icuuc
    icudata  # after: icuuc
    wavelib
    m
    tcmalloc_minimal
)

# lib: parser
add_library (parser STATIC "src/parser.cpp")
add_dependencies (parser ${ngrams_dep})
target_link_libraries (parser ${ngrams_libs})

# lib: utils
add_library (utils STATIC "src/utils.cpp")
add_dependencies (utils ${ngrams_dep})
target_link_libraries (utils ${ngrams_libs})

# executable: calc_dtw_simple
add_executable (calc_dtw_simple "src/calc_dtw_simple.cpp")
add_dependencies (calc_dtw_simple ${ngrams_dep})
target_link_libraries (
    calc_dtw_simple
    parser
    utils
    ${ngrams_libs}
)

# executable: create
add_executable (create "src/create.cpp")
add_dependencies (create ${ngrams_dep})
target_link_libraries (
    create
    parser
    utils
    ${ngrams_libs}
)

# executable: dump_wavelet_index
add_executable (dump_wavelet_index "src/dump_wavelet_index.cpp")
add_dependencies (dump_wavelet_index ${ngrams_dep})
target_link_libraries (
    dump_wavelet_index
    parser
    utils
    ${ngrams_libs}
)

# executable: normalize
add_executable (normalize "src/normalize.cpp")
add_dependencies (normalize ${ngrams_dep})
target_link_libraries (
    normalize
    parser
    utils
    ${ngrams_libs}
)

# executable: index_dtw
add_executable (index_dtw "src/index_dtw.cpp")
add_dependencies (index_dtw ${ngrams_dep})
target_link_libraries (
    index_dtw
    parser
    utils
    ${ngrams_libs}
)

# executable: index_wavelet
add_executable (index_wavelet "src/index_wavelet.cpp")
add_dependencies (index_wavelet ${ngrams_dep})
target_link_libraries (
    index_wavelet
    parser
    utils
    ${ngrams_libs}
)

# executable: print_wavelet_tree
add_executable (print_wavelet_tree "src/print_wavelet_tree.cpp")
add_dependencies (print_wavelet_tree ${ngrams_dep})
target_link_libraries (
    print_wavelet_tree
    parser
    utils
    ${ngrams_libs}
)

# executable: query_dtw_indexed
add_executable (query_dtw_indexed "src/query_dtw_indexed.cpp")
add_dependencies (query_dtw_indexed ${ngrams_dep})
target_link_libraries (
    query_dtw_indexed
    parser
    utils
    ${ngrams_libs}
)

# executable: query_dtw_simple
add_executable (query_dtw_simple "src/query_dtw_simple.cpp")
add_dependencies (query_dtw_simple ${ngrams_dep})
target_link_libraries (
    query_dtw_simple
    parser
    utils
    ${ngrams_libs}
)

# executable: scan
add_executable (scan "src/scan.cpp")
add_dependencies (scan ${ngrams_dep})
target_link_libraries (
    scan
    parser
    utils
    ${ngrams_libs}
)

# executable: store
add_executable (store "src/store.cpp")
add_dependencies (store ${ngrams_dep})
target_link_libraries (
    store
    parser
    utils
    ${ngrams_libs}
)
