#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "rocksdb/status.h"

namespace astervec {

// The engine's entire dependency on the graph storage backend, extracted to
// an interface (2026-09 batch2 storage seam). Aster/RocksGraph is the only
// implementation today (aster_graph_store.h); the seam exists so a future
// backend swap is an adapter project, not surgery — the engine talks in its
// own id type (uint64_t; bit 63 tags update-ids) and never sees backend
// types.
//
// Semantics contract (matching the Aster-backed behavior the engine was
// built against):
// - AddEdgeBatch creates vertices implicitly; forward and reverse edges are
//   whatever pairs the caller supplies (the engine inserts both directions).
// - GetOutEdges clears *out and fills the out-neighbor ids of `id`; a
//   missing vertex is NOT an error (empty result, OK status). A decode
//   failure of the stored adjacency returns Corruption.
class IGraphStore {
public:
    using Status = ROCKSDB_NAMESPACE::Status;

    virtual ~IGraphStore() = default;

    virtual Status AddVertex(uint64_t id) = 0;
    virtual Status AddEdgeBatch(
        const std::vector<std::pair<uint64_t, uint64_t>>& edges) = 0;
    virtual Status DeleteEdgeBatch(
        const std::vector<std::pair<uint64_t, uint64_t>>& edges) = 0;
    virtual Status GetOutEdges(uint64_t id, std::vector<uint64_t>* out) = 0;

    // Make every write accepted so far durable (WAL fsync). Backends with
    // no log of their own have nothing to do.
    virtual Status SyncWAL() { return Status::OK(); }
};

}  // namespace astervec
