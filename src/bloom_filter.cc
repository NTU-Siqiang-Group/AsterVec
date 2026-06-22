#include "bloom_filter.h"

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace astervec {

namespace {

constexpr double kLn2      = 0.6931471805599453;   // ln(2)
constexpr double kLn2Sq    = kLn2 * kLn2;
constexpr uint64_t kHash1Seed = 0x243f6a8885a308d3ULL;  // digits of π
constexpr uint64_t kHash2Seed = 0xb7e151628aed2a6bULL;  // digits of e

// splitmix64 finalizer — uniform, full-range, cheap.
inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

}  // namespace

std::size_t BloomFilter::bits_for(std::size_t n, double p) noexcept {
    if (n == 0) return 0;
    const double m = -static_cast<double>(n) * std::log(p) / kLn2Sq;
    // Round up; ensure at least one word.
    std::size_t bits = static_cast<std::size_t>(std::ceil(m));
    if (bits < 64) bits = 64;
    return bits;
}

std::size_t BloomFilter::hash_count_for(double p) noexcept {
    const double k = -std::log(p) / kLn2;
    std::size_t count = static_cast<std::size_t>(std::ceil(k));
    if (count < 1)  count = 1;
    if (count > 32) count = 32;
    return count;
}

uint64_t BloomFilter::hash1(uint64_t key) noexcept {
    return splitmix64(key ^ kHash1Seed);
}

uint64_t BloomFilter::hash2(uint64_t key) noexcept {
    // Ensure non-zero stride so the i-th probe never collapses onto h1 when i > 0.
    uint64_t h = splitmix64(key ^ kHash2Seed);
    return h | 1ULL;
}

BloomFilter::BloomFilter() = default;

BloomFilter::BloomFilter(std::size_t capacity, double fp_rate) {
    reset(capacity, fp_rate);
}

void BloomFilter::allocate_for(std::size_t bit_count) {
    bit_count_ = bit_count;
    const std::size_t words = (bit_count + 63) / 64;
    bits_.assign(words, 0);
}

void BloomFilter::reset(std::size_t new_capacity, double fp_rate) {
    if (fp_rate <= 0.0 || fp_rate >= 1.0) {
        throw std::invalid_argument("BloomFilter fp_rate must be in (0, 1)");
    }
    capacity_ = new_capacity;
    inserts_  = 0;
    num_hash_ = hash_count_for(fp_rate);
    allocate_for(bits_for(new_capacity, fp_rate));
}

void BloomFilter::add(uint64_t key) {
    if (bit_count_ == 0) return;
    const uint64_t h1 = hash1(key);
    const uint64_t h2 = hash2(key);
    uint64_t probe = h1;
    for (std::size_t i = 0; i < num_hash_; ++i) {
        const std::size_t bit = probe % bit_count_;
        bits_[bit >> 6] |= (1ULL << (bit & 63));
        probe += h2;
    }
    ++inserts_;
}

bool BloomFilter::might_contain(uint64_t key) const {
    if (bit_count_ == 0) return false;
    const uint64_t h1 = hash1(key);
    const uint64_t h2 = hash2(key);
    uint64_t probe = h1;
    for (std::size_t i = 0; i < num_hash_; ++i) {
        const std::size_t bit = probe % bit_count_;
        if ((bits_[bit >> 6] & (1ULL << (bit & 63))) == 0) return false;
        probe += h2;
    }
    return true;
}

namespace {
template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template <typename T>
void read_pod(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}
}  // namespace

void BloomFilter::write(std::ostream& out) const {
    const uint64_t header[4] = {
        static_cast<uint64_t>(bit_count_),
        static_cast<uint64_t>(num_hash_),
        static_cast<uint64_t>(capacity_),
        static_cast<uint64_t>(inserts_),
    };
    for (auto v : header) write_pod(out, v);
    const uint64_t words = static_cast<uint64_t>(bits_.size());
    write_pod(out, words);
    if (!bits_.empty()) {
        out.write(reinterpret_cast<const char*>(bits_.data()),
                  static_cast<std::streamsize>(bits_.size() * sizeof(uint64_t)));
    }
}

void BloomFilter::read(std::istream& in) {
    uint64_t bit_count, num_hash, capacity, inserts;
    read_pod(in, bit_count);
    read_pod(in, num_hash);
    read_pod(in, capacity);
    read_pod(in, inserts);
    uint64_t words;
    read_pod(in, words);

    bit_count_ = static_cast<std::size_t>(bit_count);
    num_hash_  = static_cast<std::size_t>(num_hash);
    capacity_  = static_cast<std::size_t>(capacity);
    inserts_   = static_cast<std::size_t>(inserts);
    bits_.assign(static_cast<std::size_t>(words), 0);
    if (words > 0) {
        in.read(reinterpret_cast<char*>(bits_.data()),
                static_cast<std::streamsize>(words * sizeof(uint64_t)));
    }
}

}  // namespace astervec
