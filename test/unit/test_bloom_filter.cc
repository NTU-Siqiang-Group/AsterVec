#include <sstream>

#include "bloom_filter.h"
#include "doctest.h"

using lsm_vec::BloomFilter;

TEST_CASE("BloomFilter: empty filter reports nothing contained") {
    BloomFilter bf;  // default-constructed (capacity 0)
    for (uint64_t k = 0; k < 1000; ++k) {
        CHECK_FALSE(bf.might_contain(k));
    }
    CHECK(bf.size() == 0);
    CHECK(bf.fill_ratio() == 0.0);
}

TEST_CASE("BloomFilter: after add, might_contain returns true for every added key") {
    BloomFilter bf(1000, 0.01);
    for (uint64_t k = 0; k < 1000; ++k) bf.add(k);
    for (uint64_t k = 0; k < 1000; ++k) {
        CHECK(bf.might_contain(k));
    }
    CHECK(bf.size() == 1000);
}

TEST_CASE("BloomFilter: FP rate at design capacity within slack of target") {
    // With n=10000 design and fp=0.01 target, measuring FPs on 10000 unseen
    // keys should stay under ~2% (factor-2 slack for randomness).
    BloomFilter bf(10000, 0.01);
    for (uint64_t k = 0; k < 10000; ++k) bf.add(k);
    int fp = 0;
    for (uint64_t k = 100000; k < 110000; ++k) {
        if (bf.might_contain(k)) ++fp;
    }
    CHECK(fp < 200);  // 2% of 10000
}

TEST_CASE("BloomFilter: fill_ratio is monotonic non-decreasing") {
    BloomFilter bf(1000, 0.01);
    double prev = bf.fill_ratio();
    for (uint64_t k = 0; k < 500; ++k) {
        bf.add(k);
        const double cur = bf.fill_ratio();
        CHECK(cur >= prev);
        prev = cur;
    }
    CHECK(prev > 0.0);
}

TEST_CASE("BloomFilter: reset to larger capacity and re-add survives all probes") {
    BloomFilter bf(500, 0.01);
    for (uint64_t k = 0; k < 100; ++k) bf.add(k);
    CHECK(bf.capacity() == 500);

    bf.reset(5000, 0.01);
    CHECK(bf.capacity() == 5000);
    CHECK(bf.size() == 0);
    for (uint64_t k = 0; k < 100; ++k) CHECK_FALSE(bf.might_contain(k));  // freshly empty

    for (uint64_t k = 0; k < 100; ++k) bf.add(k);
    for (uint64_t k = 0; k < 100; ++k) CHECK(bf.might_contain(k));
}

TEST_CASE("BloomFilter: write/read round-trip preserves membership") {
    BloomFilter src(256, 0.01);
    for (uint64_t k = 0; k < 50; ++k) src.add(k * 13 + 7);

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    src.write(ss);

    BloomFilter dst;
    dst.read(ss);
    CHECK(dst.capacity() == src.capacity());
    CHECK(dst.size()     == src.size());
    for (uint64_t k = 0; k < 50; ++k) {
        CHECK(dst.might_contain(k * 13 + 7));
    }
}

TEST_CASE("BloomFilter: invalid fp_rate throws") {
    CHECK_THROWS_AS(BloomFilter(100, 0.0),  std::invalid_argument);
    CHECK_THROWS_AS(BloomFilter(100, 1.0),  std::invalid_argument);
    CHECK_THROWS_AS(BloomFilter(100, -0.5), std::invalid_argument);
}
