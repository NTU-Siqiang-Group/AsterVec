CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2

REPO_ROOT := ..
ASTER_ROOT := $(REPO_ROOT)/lib/aster
BUILD_DIR ?= $(REPO_ROOT)/build

ASTERVEC_STATIC := $(BUILD_DIR)/lib/libastervec.a
TEST_SOURCES := test.cc

INCLUDES := \
	-I$(REPO_ROOT)/include \
	-I$(ASTER_ROOT)/include

LIB_DIRS := \
	-L$(ASTER_ROOT) \
	-L$(ASTER_ROOT)/build \
	-L$(ASTER_ROOT)/build/lib \
	-L$(ASTER_ROOT)/lib

LIBS := -lrocksdb -lzstd -lsnappy -llz4 -lbz2 -lz -lpthread -ldl -ljemalloc

TARGET := astervec_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(ASTERVEC_STATIC) $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_DIRS) $(TEST_SOURCES) $(ASTERVEC_STATIC) -o $@ $(LIBS)

$(ASTERVEC_STATIC):
	$(MAKE) -C $(REPO_ROOT) static

clean:
	rm -f $(TARGET)
