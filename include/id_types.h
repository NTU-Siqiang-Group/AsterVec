#pragma once

#include <cstdint>

namespace lsm_vec {

// User-facing identifier. Must satisfy real_id <= kMaxRealId.
using real_id_t = uint64_t;

// HNSW / edge-store / vector-store identifier.
// Bit 63 discriminates:
//   0 => internal_id is identical to its owning real_id (direct mapping)
//   1 => internal_id was allocated by Update / Delete-then-Insert
//        (sparse maps carry its real_id)
using internal_id_t = uint64_t;

constexpr internal_id_t kUpdateIdBit   = 1ULL << 63;
constexpr internal_id_t kFirstUpdateId = kUpdateIdBit;
constexpr real_id_t     kMaxRealId     = kUpdateIdBit - 1;

constexpr bool is_direct_id(internal_id_t i) noexcept { return (i >> 63) == 0; }
constexpr bool is_update_id(internal_id_t i) noexcept { return (i >> 63) == 1; }

// Offset within the update-id range. Only meaningful when is_update_id(i).
constexpr internal_id_t update_id_index(internal_id_t i) noexcept {
    return i - kFirstUpdateId;
}

static_assert(is_direct_id(0));
static_assert(is_direct_id(kMaxRealId));
static_assert(is_update_id(kFirstUpdateId));
static_assert(is_update_id(~0ULL));
static_assert(update_id_index(kFirstUpdateId) == 0);
static_assert(update_id_index(kFirstUpdateId + 42) == 42);
static_assert(kMaxRealId + 1 == kFirstUpdateId);

}  // namespace lsm_vec
