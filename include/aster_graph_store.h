#pragma once

#include <memory>
#include <string>

#include "graph_store.h"
#include "rocksdb/graph.h"

namespace astervec {

// The Aster (RocksGraph) implementation of IGraphStore — the ONLY place in
// the engine that touches RocksGraph or Aster's edge types. Conversion
// boundary: the engine's uint64_t ids become Aster's int64_t node_id_t here
// (two's-complement casts; bit-63 update-ids round-trip exactly, same as the
// engine's previous direct calls did).
//
// Constructing the store IS opening the graph; the direct-IO reopen path
// simply constructs a fresh AsterGraphStore over the same path with the
// caller's (retuned) options, exactly as the engine previously rebuilt
// RocksGraph in place.
class AsterGraphStore final : public IGraphStore {
public:
    AsterGraphStore(rocksdb::Options& options, int edge_update_policy,
                    uint32_t encoding_type, bool auto_reinitialize,
                    const std::string& db_path, bool minimal_mode,
                    bool skip_property_cfs);

    Status AddVertex(uint64_t id) override;
    Status AddEdgeBatch(
        const std::vector<std::pair<uint64_t, uint64_t>>& edges) override;
    Status DeleteEdgeBatch(
        const std::vector<std::pair<uint64_t, uint64_t>>& edges) override;
    Status GetOutEdges(uint64_t id, std::vector<uint64_t>* out) override;

private:
    // No conversion scratch member: batch calls may run concurrently (the
    // engine has concurrent insert paths), so conversion uses a local vector
    // per call. RocksGraph itself is thread-safe for these operations.
    std::unique_ptr<rocksdb::RocksGraph> graph_;
};

}  // namespace astervec
