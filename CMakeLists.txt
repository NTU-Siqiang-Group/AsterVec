# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(LSMVEC)

# Set C++ standard to C++17 and build type
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O2")

# On Apple Silicon, Homebrew installs to /opt/homebrew instead of /usr/local.
# Add it to the search paths so find_package/find_library can locate dependencies.
if(APPLE AND EXISTS /opt/homebrew)
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

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
    message(FATAL_ERROR "Aster RocksDB library or headers not found")
endif()

message(STATUS "Using RocksDB library at ${ROCKSDB_LIB}")
message(STATUS "Using RocksDB headers at ${ROCKSDB_INCLUDE_DIR}")

include_directories(${ROCKSDB_INCLUDE_DIR})

# find_library(ROCKSDB_LIB rocksdb HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
# set(ROCKSDB_INCLUDE_DIR /usr/include/rocksdb)

if(NOT ROCKSDB_LIB OR NOT ROCKSDB_INCLUDE_DIR)
    message(FATAL_ERROR "RocksDB library or headers not found")
endif()

message(STATUS "Found RocksDB library at ${ROCKSDB_LIB}")
message(STATUS "Found RocksDB headers at ${ROCKSDB_INCLUDE_DIR}")

include_directories(${ROCKSDB_INCLUDE_DIR})

# Specify the source files
set(LIB_SOURCES
    src/lsm_vec_index.cc
    src/lsm_vec_db.cc
    src/utils.cc
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
        message(WARNING "jemalloc not found, trying manual link...")
        target_link_libraries(lsmvec_shared PRIVATE jemalloc)
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

option(LSMVEC_BUILD_PYTHON "Build Python bindings" OFF)

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
    target_link_libraries(lsm_vec_python PRIVATE lsmvec_shared)
    set_target_properties(lsm_vec_python PROPERTIES
        OUTPUT_NAME lsm_vec
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    install(TARGETS lsm_vec_python LIBRARY DESTINATION .)
endif()
