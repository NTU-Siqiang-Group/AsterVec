.PHONY: all configure static shared lib bin clean aster unit_test \
        unit_test_asan unit_test_tsan unit_test_ubsan

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CMAKE ?= cmake
DOCTEST_TAGS ?=

all: lib bin

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

static: configure
	$(CMAKE) --build $(BUILD_DIR) --target astervec_static -j

shared: configure
	$(CMAKE) --build $(BUILD_DIR) --target astervec_shared -j

lib: static shared

bin: configure
	$(CMAKE) --build $(BUILD_DIR) --target astervec -j

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Build Aster (RocksDB) with only ZSTD compression. snappy/lz4/bzip/zlib are
# disabled so the build needs just libzstd (build_detect_platform reads these
# env vars). NOTE: make_config.mk is cached — if you previously built Aster with
# those libs, run `rm -f lib/aster/make_config.mk` (or `make -C lib/aster clean`)
# before `make aster` so the platform is re-detected.
aster:
	ROCKSDB_DISABLE_SNAPPY=1 ROCKSDB_DISABLE_LZ4=1 ROCKSDB_DISABLE_BZIP=1 ROCKSDB_DISABLE_ZLIB=1 \
	ROCKSDB_DISABLE_JEMALLOC=1 \
	$(MAKE) -C lib/aster static_lib -j$(NPROC) DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 EXTRA_CXXFLAGS=-fPIC

clean:
	rm -rf $(BUILD_DIR)

unit_test: configure
	$(CMAKE) --build $(BUILD_DIR) --target astervec_unit_tests -- -j
	$(BUILD_DIR)/test/unit/astervec_unit_tests $(DOCTEST_TAGS)

# Sanitizer-instrumented unit-test runs. Each uses a separate build dir
# so it doesn't clobber the Release build. Concurrent-writer-refactor
# Phase 0 deliverable: pure-AsterVec sanitizer target (Aster is built
# via its own Makefile and is not instrumented — by design).
unit_test_asan:
	$(CMAKE) -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DASTERVEC_SANITIZER=asan
	$(CMAKE) --build build-asan --target astervec_unit_tests -- -j
	build-asan/test/unit/astervec_unit_tests $(DOCTEST_TAGS)

unit_test_tsan:
	$(CMAKE) -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DASTERVEC_SANITIZER=tsan
	$(CMAKE) --build build-tsan --target astervec_unit_tests -- -j
	build-tsan/test/unit/astervec_unit_tests $(DOCTEST_TAGS)

unit_test_ubsan:
	$(CMAKE) -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DASTERVEC_SANITIZER=ubsan
	$(CMAKE) --build build-ubsan --target astervec_unit_tests -- -j
	build-ubsan/test/unit/astervec_unit_tests $(DOCTEST_TAGS)
