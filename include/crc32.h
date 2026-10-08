#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace astervec {
namespace crc32 {

// Reflected CRC-32 (poly 0xEDB88320), table driven.
// Raw-register form: start from kInit, feed bytes with Update(), and finish
// with Finish(). Value() is the one-shot form.
constexpr uint32_t kInit = 0xFFFFFFFFu;

inline uint32_t Update(uint32_t crc, const void* data, size_t n)
{
    // Magic-static IIFE: thread-safe one-time init (a hand-rolled
    // `static bool init` guard is a data race under concurrent first
    // calls — caught by TSan in review R4).
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++)
        crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

inline uint32_t Finish(uint32_t crc) { return crc ^ 0xFFFFFFFFu; }

inline uint32_t Value(const void* data, size_t n)
{
    return Finish(Update(kInit, data, n));
}

}  // namespace crc32
}  // namespace astervec
