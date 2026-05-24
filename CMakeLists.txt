# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(LSMVEC)

# Set C++ standard to C++17 and build type
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O2")

# Sanitizer build target — concurrent-writer-refactor-plan §6 Phase 0.
# LSMVEC_SANITIZER=OFF | asan | tsan | ubsan
# Use as: cmake -B build-tsan -DLSMVEC_SANITIZER=tsan
# Implies -O1 -g and frame pointers; intended for the unit-test target only.
# Aster (lib/aster) is built separately via its own Makefile and does NOT
# pick up these flags — that's deliberate (Aster has known bad-free in its
# write_stall_stats static initializer that ASan/TSan flags as a false
# positive on the prebuilt static archive; see test_concurrent_search.cc
# header note).
set(LSMVEC_SANITIZER "OFF" CACHE STRING "Enable a sanitizer build: OFF|asan|tsan|ubsan")
set_property(CACHE LSMVEC_SANITIZER PROPERTY STRINGS OFF asan tsan ubsan)
if(NOT LSMVEC_SANITIZER STREQUAL "OFF")
    set(_lsmvec_san_flags "-fno-omit-frame-pointer -g -O1")
    if(LSMVEC_SANITIZER STREQUAL "asan")
        set(_lsmvec_san_flags "${_lsmvec_san_flags} -fsanitize=address")
    elseif(LSMVEC_SANITIZER STREQUAL "tsan")
        set(_lsmvec_san_flags "${_lsmvec_san_flags} -fsanitize=thread")
    elseif(LSMVEC_SANITIZER STREQUAL "ubsan")
        set(_lsmvec_san_flags "${_lsmvec_san_flags} -fsanitize=undefined")
    else()
        message(FATAL_ERROR "LSMVEC_SANITIZER must be one of OFF|asan|tsan|ubsan, got: ${LSMVEC_SANITIZER}")
    endif()
    message(STATUS "Sanitizer enabled: ${LSMVEC_SANITIZER} (${_lsmvec_san_flags})")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_lsmvec_san_flags}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${_lsmvec_san_flags}")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${_lsmvec_san_flags}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_lsmvec_san_flags}")
endif()

# SIMD flags: AVX2/FMA on x86_64 only (ARM uses scalar fallback in distance.h)
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    check_cxx_compiler_flag("-mfma" HAS_FMA)
    if(HAS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    endif()
    if(HAS_FMA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    endif()
endif()

# On Apple Silicon, Homebrew installs to /opt/homebrew instead of /usr/local.
# Add it to the search paths so find_package/find_library can locate dependencies.
if(APPLE AND EXISTS /opt/homebrew)
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party/json)

find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()
add_compile_definitions(BOOST_RANDOM_USE_STD_RANDOM ROCKSDB_USE_STD_RANDOM)

set(ASTER_ROCKSDB_ROOT ${CMAKE_SOURCE_DIR}/lib/aster)

find_library(ROCKSDB_LIB rocksdb
    HINTS
        "${ASTER_ROCKSDB_ROOT}"
        "${ASTER_ROCKSDB_ROOT}/build"
        "${ASTER_ROCKSDB_ROOT}/build/lib"
        "${ASTER_ROCKSDB_ROOT}/lib"
)

find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h
    HINTS
        "${ASTER_ROCKSDB_ROOT}/include"
        "${ASTER_ROCKSDB_ROOT}"
)

if(NOT ROCKSDB_LIB OR NOT ROCKSDB_INCLUDE_DIR)
    message(STATUS "Pre-built Aster not found, building from source...")
    include(ProcessorCount)
    ProcessorCount(NPROC)
    if(NPROC EQUAL 0)
        set(NPROC 4)
    endif()
    execute_process(
        COMMAND make static_lib -j${NPROC} DEBUG_LEVEL=0
                DISABLE_WARNING_AS_ERROR=1 EXTRA_CXXFLAGS=-fPIC
        WORKING_DIRECTORY "${ASTER_ROCKSDB_ROOT}"
        RESULT_VARIABLE ASTER_BUILD_RESULT
    )
    if(NOT ASTER_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build Aster from source")
    endif()
    # Re-find after build
    find_library(ROCKSDB_LIB rocksdb
        HINTS "${ASTER_ROCKSDB_ROOT}" "${ASTER_ROCKSDB_ROOT}/lib"
    )
    find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h
        HINTS "${ASTER_ROCKSDB_ROOT}/include"
    )
    if(NOT ROCKSDB_LIB OR NOT ROCKSDB_INCLUDE_DIR)
        message(FATAL_ERROR "Aster build succeeded but library/headers still not found")
    endif()
endif()

message(STATUS "Using RocksDB library at ${ROCKSDB_LIB}")
message(STATUS "Using RocksDB headers at ${ROCKSDB_INCLUDE_DIR}")

include_directories(${ROCKSDB_INCLUDE_DIR})

# Specify the source files
set(LIB_SOURCES
    src/lsm_vec_index.cc
    src/lsm_vec_db.cc
    src/utils.cc
    src/metadata.cc
    src/metadata_store.cc
    src/bloom_filter.cc
    src/lsm_vec_rnn_descent.cc
    src/lsm_vec_bulk_build.cc
)

find_library(ZSTD_LIB zstd
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
          /opt/homebrew/lib
)

if(NOT ZSTD_LIB)
    message(FATAL_ERROR "libzstd not found; please install libzstd-dev or provide its path")
endif()

message(STATUS "Using ZSTD library at ${ZSTD_LIB}")

find_library(SNAPPY_LIB snappy
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/opt/snappy/lib
          /opt/homebrew/lib /opt/homebrew/opt/snappy/lib
)
find_library(LZ4_LIB lz4
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/opt/lz4/lib
          /opt/homebrew/lib /opt/homebrew/opt/lz4/lib
)
find_library(BZ2_LIB bz2
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/opt/bzip2/lib
          /opt/homebrew/lib /opt/homebrew/opt/bzip2/lib
)

if(NOT SNAPPY_LIB)
    message(FATAL_ERROR "libsnappy not found; please install snappy or provide its path")
endif()
if(NOT LZ4_LIB)
    message(FATAL_ERROR "liblz4 not found; please install lz4 or provide its path")
endif()
if(NOT BZ2_LIB)
    message(FATAL_ERROR "libbz2 not found; please install bzip2 or provide its path")
endif()

message(STATUS "Using Snappy library at ${SNAPPY_LIB}")
message(STATUS "Using LZ4 library at ${LZ4_LIB}")
message(STATUS "Using BZ2 library at ${BZ2_LIB}")

set(LSMVEC_LINK_LIBS
    ${ROCKSDB_LIB}
    ${ZSTD_LIB}
    ${SNAPPY_LIB}
    ${LZ4_LIB}
    ${BZ2_LIB}
    z
    pthread
    dl
)

# Add the library targets
add_library(lsmvec_obj OBJECT ${LIB_SOURCES})
target_include_directories(lsmvec_obj PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(lsmvec_static STATIC $<TARGET_OBJECTS:lsmvec_obj>)
add_library(lsmvec_shared SHARED $<TARGET_OBJECTS:lsmvec_obj>)

target_include_directories(lsmvec_static PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(lsmvec_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(lsmvec_static PUBLIC ${LSMVEC_LINK_LIBS})
target_link_libraries(lsmvec_shared PUBLIC ${LSMVEC_LINK_LIBS})
if(APPLE)
    find_library(JEMALLOC_LIB NAMES jemalloc
        HINTS /usr/local/lib /opt/homebrew/lib /opt/homebrew/opt/jemalloc/lib
    )

    if(JEMALLOC_LIB)
        message(STATUS "Found jemalloc at: ${JEMALLOC_LIB}")
        target_link_libraries(lsmvec_shared PRIVATE ${JEMALLOC_LIB})
    else()
        message(STATUS "jemalloc not found, skipping (optional)")
    endif()
endif()

set_target_properties(lsmvec_static PROPERTIES
    OUTPUT_NAME lsmvec
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(lsmvec_shared PROPERTIES
    OUTPUT_NAME lsmvec
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add the executable target
add_executable(lsm_vec test/test.cc)
target_include_directories(lsm_vec PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lsm_vec PRIVATE lsmvec_shared)

# Set output directory for the executable
set_target_properties(lsm_vec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Concurrent search benchmark tool — for Phase A multi-thread perf testing.
add_executable(concurrent_search_bench tools/concurrent_search_bench.cc)
target_include_directories(concurrent_search_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(concurrent_search_bench PRIVATE lsmvec_shared)
set_target_properties(concurrent_search_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

option(LSMVEC_BUILD_HTTP "Build the HTTP/REST server (lsm_vec_http)" ON)

if(LSMVEC_BUILD_HTTP)
    # cpp-httplib is header-only, vendored at third_party/httplib/.
    add_executable(lsm_vec_http
        src/lsm_vec_http_main.cc
        src/lsm_vec_http.cc
    )
    target_include_directories(lsm_vec_http PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/httplib
        ${CMAKE_SOURCE_DIR}/third_party/json
    )
    target_link_libraries(lsm_vec_http PRIVATE lsmvec_shared)
    # httplib requires threads.
    find_package(Threads REQUIRED)
    target_link_libraries(lsm_vec_http PRIVATE Threads::Threads)
    set_target_properties(lsm_vec_http PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

option(LSMVEC_BUILD_PYTHON "Build Python bindings" OFF)
option(LSMVEC_PYTHON_STATIC "Link Python module against static lib (for wheel builds)" OFF)

if(LSMVEC_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(lsm_vec_python MODULE python/lsm_vec_pybind.cc)
    target_include_directories(lsm_vec_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(LSMVEC_PYTHON_STATIC)
        target_link_libraries(lsm_vec_python PRIVATE lsmvec_static)
    else()
        target_link_libraries(lsm_vec_python PRIVATE lsmvec_shared)
    endif()
    set_target_properties(lsm_vec_python PROPERTIES
        OUTPUT_NAME lsm_vec
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    install(TARGETS lsm_vec_python LIBRARY DESTINATION .)
endif()

option(LSMVEC_BUILD_UNIT_TESTS "Build doctest-based unit tests" ON)
if(LSMVEC_BUILD_UNIT_TESTS)
    add_subdirectory(test/unit)
endif()
