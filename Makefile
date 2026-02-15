.PHONY: all configure static shared lib bin clean aster

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CMAKE ?= cmake

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