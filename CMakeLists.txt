# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ASTERVEC)

# Set C++ standard to C++17 and build type
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O2")

# Sanitizer build target — concurrent-writer-refactor-plan §6 Phase 0.
# ASTERVEC_SANITIZER=OFF | asan | tsan | ubsan
# Use as: cmake -B build-tsan -DASTERVEC_SANITIZER=tsan
# Implies -O1 -g and frame pointers; intended for the unit-test target only.
# Aster (lib/aster) is built separately via its own Makefile and does NOT
# pick up these flags — that's deliberate (Aster has known bad-free in its
# write_stall_stats static initializer that ASan/TSan flags as a false
# positive on the prebuilt static archive; see test_concurrent_search.cc
# header note).
set(ASTERVEC_SANITIZER "OFF" CACHE STRING "Enable a sanitizer build: OFF|asan|tsan|ubsan")
set_property(CACHE ASTERVEC_SANITIZER PROPERTY STRINGS OFF asan tsan ubsan)
if(NOT ASTERVEC_SANITIZER STREQUAL "OFF")
    set(_astervec_san_flags "-fno-omit-frame-pointer -g -O1")
    if(ASTERVEC_SANITIZER STREQUAL "asan")
        set(_astervec_san_flags "${_astervec_san_flags} -fsanitize=address")
    elseif(ASTERVEC_SANITIZER STREQUAL "tsan")
        set(_astervec_san_flags "${_astervec_san_flags} -fsanitize=thread")
    elseif(ASTERVEC_SANITIZER STREQUAL "ubsan")
        set(_astervec_san_flags "${_astervec_san_flags} -fsanitize=undefined")
    else()
        message(FATAL_ERROR "ASTERVEC_SANITIZER must be one of OFF|asan|tsan|ubsan, got: ${ASTERVEC_SANITIZER}")
    endif()
    message(STATUS "Sanitizer enabled: ${ASTERVEC_SANITIZER} (${_astervec_san_flags})")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_astervec_san_flags}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${_astervec_san_flags}")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${_astervec_san_flags}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_astervec_san_flags}")
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
        # Must match the top-level `make aster`: build Aster with only ZSTD
        # (snappy/lz4/bzip/zlib disabled) so librocksdb.a doesn't reference
        # codecs we no longer link. Used by the pip / direct-cmake path when
        # `make aster` wasn't run first.
        COMMAND ${CMAKE_COMMAND} -E env
                ROCKSDB_DISABLE_SNAPPY=1 ROCKSDB_DISABLE_LZ4=1
                ROCKSDB_DISABLE_BZIP=1 ROCKSDB_DISABLE_ZLIB=1
                make static_lib -j${NPROC} DEBUG_LEVEL=0
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
    src/astervec_index.cc
    src/astervec_db.cc
    src/utils.cc
    src/metadata.cc
    src/metadata_store.cc
    src/bloom_filter.cc
    src/astervec_rnn_descent.cc
    src/astervec_bulk_build.cc
)

find_library(ZSTD_LIB zstd
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
          /opt/homebrew/lib
)

if(NOT ZSTD_LIB)
    message(FATAL_ERROR "libzstd not found; please install libzstd-dev or provide its path")
endif()

message(STATUS "Using ZSTD library at ${ZSTD_LIB}")

# snappy / lz4 / bz2 / zlib are intentionally NOT required. Aster is built with
# ROCKSDB_DISABLE_{SNAPPY,LZ4,BZIP,ZLIB}=1 (see the `aster` target in Makefile),
# so librocksdb.a references none of those codecs. The only compression the
# engine uses is ZSTD (the metadata/payload column family); the graph store is
# kNoCompression. Keeping the build to a single compression dependency (zstd)
# makes it far simpler to set up.

set(ASTERVEC_LINK_LIBS
    ${ROCKSDB_LIB}
    ${ZSTD_LIB}
    pthread
    dl
)

# Add the library targets
add_library(astervec_obj OBJECT ${LIB_SOURCES})
target_include_directories(astervec_obj PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(astervec_static STATIC $<TARGET_OBJECTS:astervec_obj>)
add_library(astervec_shared SHARED $<TARGET_OBJECTS:astervec_obj>)

target_include_directories(astervec_static PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(astervec_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(astervec_static PUBLIC ${ASTERVEC_LINK_LIBS})
target_link_libraries(astervec_shared PUBLIC ${ASTERVEC_LINK_LIBS})
# NOTE: do NOT link jemalloc into astervec_shared / the embeddable libraries.
# As an embedded library (Python extension, or a C++ app linking libastervec),
# jemalloc registers a malloc zone and corrupts the host process's heap when its
# allocations interleave with the host allocator (e.g. Python + SQLite) on macOS.
# Aster is also built with ROCKSDB_DISABLE_JEMALLOC=1 (see Makefile). If the
# standalone server ever wants jemalloc, link it onto the astervec_http executable
# (process-level interposition, Linux), never here.
# See docs/local/2026-06-28_1318_JEMALLOC_EMBED_FIX.md

set_target_properties(astervec_static PROPERTIES
    OUTPUT_NAME astervec
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(astervec_shared PROPERTIES
    OUTPUT_NAME astervec
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add the executable target
add_executable(astervec test/test.cc)
target_include_directories(astervec PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(astervec PRIVATE astervec_static)

# Set output directory for the executable
set_target_properties(astervec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Concurrent search benchmark tool — for Phase A multi-thread perf testing.
add_executable(concurrent_search_bench test/concurrent_search_bench.cc)
target_include_directories(concurrent_search_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(concurrent_search_bench PRIVATE astervec_static)
set_target_properties(concurrent_search_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

option(ASTERVEC_BUILD_HTTP "Build the HTTP/REST server (astervec_http)" ON)

if(ASTERVEC_BUILD_HTTP)
    # cpp-httplib is header-only, vendored at third_party/httplib/.
    add_executable(astervec_http
        src/astervec_http_main.cc
        src/astervec_http.cc
    )
    target_include_directories(astervec_http PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/httplib
        ${CMAKE_SOURCE_DIR}/third_party/json
    )
    target_link_libraries(astervec_http PRIVATE astervec_static)
    # httplib requires threads.
    find_package(Threads REQUIRED)
    target_link_libraries(astervec_http PRIVATE Threads::Threads)
    set_target_properties(astervec_http PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

option(ASTERVEC_BUILD_PYTHON "Build Python bindings" OFF)
option(ASTERVEC_PYTHON_STATIC "Link the Python module against the static engine lib (default ON: self-contained ext, no libastervec.dylib at runtime)" ON)

if(ASTERVEC_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(astervec_python MODULE python/astervec_pybind.cc)
    target_include_directories(astervec_python PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(ASTERVEC_PYTHON_STATIC)
        target_link_libraries(astervec_python PRIVATE astervec_static)
    else()
        target_link_libraries(astervec_python PRIVATE astervec_shared)
    endif()
    set_target_properties(astervec_python PROPERTIES
        OUTPUT_NAME astervec
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    install(TARGETS astervec_python LIBRARY DESTINATION .)
endif()

option(ASTERVEC_BUILD_UNIT_TESTS "Build doctest-based unit tests" ON)
if(ASTERVEC_BUILD_UNIT_TESTS)
    add_subdirectory(test/unit)
endif()
