#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace astervec {

// Minimal additive Bloom filter sized for an expected item count and a target
// false-positive rate. Does not support removal. On overflow, call reset() with
// a larger capacity and re-add the surviving keys (this is how the update-id
// tracker grows — see DELETE_DESIGN.md §3.4).
class BloomFilter {
  public:
    // Default-constructed filter has zero capacity; all lookups return false.
    BloomFilter();

    // Precondition: capacity > 0, 0 < fp_rate < 1.
    explicit BloomFilter(std::size_t capacity, double fp_rate = 0.01);

    void add(uint64_t key);
    bool might_contain(uint64_t key) const;

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t size()     const noexcept { return inserts_;  }
    double      fill_ratio() const noexcept {
        return capacity_ == 0 ? 0.0
                              : static_cast<double>(inserts_) / capacity_;
    }

    // Re-initialize the filter empty with new capacity. Caller re-adds entries.
    void reset(std::size_t new_capacity, double fp_rate = 0.01);

    // Binary serialization (little-endian). V1 does not persist the filter
    // (it is rebuilt from updated_real_to_internal_ on Open), but the API is
    // provided so V2 can opt in without a format change.
    void write(std::ostream& out) const;
    void read(std::istream& in);

  private:
    std::vector<uint64_t> bits_;      // bit array as 64-bit words
    std::size_t           bit_count_ = 0;  // total bits m
    std::size_t           num_hash_  = 0;  // number of hash probes k
    std::size_t           capacity_  = 0;  // design item count n
    std::size_t           inserts_   = 0;  // count of add() calls

    static std::size_t bits_for(std::size_t n, double p) noexcept;
    static std::size_t hash_count_for(double p) noexcept;
    static uint64_t    hash1(uint64_t key) noexcept;
    static uint64_t    hash2(uint64_t key) noexcept;

    void allocate_for(std::size_t bit_count);
};

}  // namespace astervec
