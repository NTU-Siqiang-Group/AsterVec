#include "aster_graph_store.h"

namespace astervec {

namespace {

std::vector<std::pair<rocksdb::node_id_t, rocksdb::node_id_t>> ToAster(
    const std::vector<std::pair<uint64_t, uint64_t>>& edges)
{
    std::vector<std::pair<rocksdb::node_id_t, rocksdb::node_id_t>> out;
    out.reserve(edges.size());
    for (const auto& [from, to] : edges) {
        out.emplace_back(static_cast<rocksdb::node_id_t>(from),
                         static_cast<rocksdb::node_id_t>(to));
    }
    return out;
}

}  // namespace

AsterGraphStore::AsterGraphStore(rocksdb::Options& options,
                                 int edge_update_policy,
                                 uint32_t encoding_type,
                                 bool auto_reinitialize,
                                 const std::string& db_path,
                                 bool minimal_mode,
                                 bool skip_property_cfs)
    : graph_(std::make_unique<rocksdb::RocksGraph>(
          options, edge_update_policy, static_cast<int>(encoding_type),
          auto_reinitialize, db_path, minimal_mode, skip_property_cfs))
{
}

IGraphStore::Status AsterGraphStore::AddVertex(uint64_t id)
{
    return graph_->AddVertex(static_cast<rocksdb::node_id_t>(id));
}

IGraphStore::Status AsterGraphStore::AddEdgeBatch(
    const std::vector<std::pair<uint64_t, uint64_t>>& edges)
{
    return graph_->AddEdgeBatch(ToAster(edges));
}

IGraphStore::Status AsterGraphStore::DeleteEdgeBatch(
    const std::vector<std::pair<uint64_t, uint64_t>>& edges)
{
    return graph_->DeleteEdgeBatch(ToAster(edges));
}

IGraphStore::Status AsterGraphStore::GetOutEdges(uint64_t id,
                                                 std::vector<uint64_t>* out)
{
    out->clear();
    rocksdb::Edges edges;
    Status s =
        graph_->GetAllEdges(static_cast<rocksdb::node_id_t>(id), &edges);
    if (s.IsNotFound()) {
        return Status::OK();  // missing vertex reads as empty adjacency
    }
    if (!s.ok()) {
        return s;  // incl. Corruption from a failed adjacency decode
    }
    out->reserve(edges.num_edges_out);
    for (uint32_t i = 0; i < edges.num_edges_out; ++i) {
        out->push_back(static_cast<uint64_t>(edges.nxts_out[i].nxt));
    }
    rocksdb::free_edges(&edges);
    return Status::OK();
}

}  // namespace astervec
