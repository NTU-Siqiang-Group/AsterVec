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
	$(CMAKE) --build $(BUILD_DIR) --target lsmvec_static -j

shared: configure
	$(CMAKE) --build $(BUILD_DIR) --target lsmvec_shared -j

lib: static shared

bin: configure
	$(CMAKE) --build $(BUILD_DIR) --target lsm_vec -j

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

aster:
	$(MAKE) -C lib/aster static_lib -j$(NPROC) DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 EXTRA_CXXFLAGS=-fPIC

clean:
	rm -rf $(BUILD_DIR)

unit_test: configure
	$(CMAKE) --build $(BUILD_DIR) --target lsmvec_unit_tests -- -j
	$(BUILD_DIR)/test/unit/lsmvec_unit_tests $(DOCTEST_TAGS)

# Sanitizer-instrumented unit-test runs. Each uses a separate build dir
# so it doesn't clobber the Release build. Concurrent-writer-refactor
# Phase 0 deliverable: pure-LSM-Vec sanitizer target (Aster is built
# via its own Makefile and is not instrumented — by design).
unit_test_asan:
	$(CMAKE) -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DLSMVEC_SANITIZER=asan
	$(CMAKE) --build build-asan --target lsmvec_unit_tests -- -j
	build-asan/test/unit/lsmvec_unit_tests $(DOCTEST_TAGS)

unit_test_tsan:
	$(CMAKE) -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DLSMVEC_SANITIZER=tsan
	$(CMAKE) --build build-tsan --target lsmvec_unit_tests -- -j
	build-tsan/test/unit/lsmvec_unit_tests $(DOCTEST_TAGS)

unit_test_ubsan:
	$(CMAKE) -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DLSMVEC_SANITIZER=ubsan
	$(CMAKE) --build build-ubsan --target lsmvec_unit_tests -- -j
	build-ubsan/test/unit/lsmvec_unit_tests $(DOCTEST_TAGS)
