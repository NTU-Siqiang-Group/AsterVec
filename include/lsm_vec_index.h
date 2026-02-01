#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <string>
#include "rocksdb/graph.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include "rocksdb/status.h"
#include <iostream>
#include "disk_vector.h"
#include "lsm_vec_db.h"
#include "statistics.h"
#include "logger.h"

namespace lsm_vec
{
using namespace ROCKSDB_NAMESPACE;

    class LSMVec
    {
    public:
        struct Node
        {
            node_id_t id;
            std::vector<float> point;
            std::unordered_map<int, std::vector<node_id_t>> neighbors; // Layer -> neighbors
        };

        mutable HNSWStats stats;

        bool use_heuristic_neighbor_selection_ = true;

        void setNeighborSelection(bool useHeuristic) {
            use_heuristic_neighbor_selection_ = useHeuristic;
        }

        LSMVec(const std::string& db_path,
               const LSMVecDBOptions& options,
               std::ostream &outFile);

        Status SerializeMetadata(std::ostream& out) const;
        Status DeserializeMetadata(std::istream& in);
        const std::unordered_set<node_id_t>& deletedIds() const { return deleted_ids_; }
        void setDeletedIds(const std::unordered_set<node_id_t>& ids) { deleted_ids_ = ids; }

        void insertNode(node_id_t nodeId, const std::vector<float> &vector);
        node_id_t knnSearch(const std::vector<float> &queryVector);
        Status deleteNode(node_id_t id);
        Status updateNode(node_id_t id, const std::vector<float>& vec);
        Status getNodeVector(node_id_t id, std::vector<float>* out);
        std::vector<SearchResult> knnSearchK(const std::vector<float>& query, int k, int ef_search);
        std::unordered_set<int> highest_layer_nodes_;

        void printIndexStatus() const;
        void printStatistics() const;
        void printState() const;
        void close();

        std::string vector_file_path_;
        int vector_dim_ = 0;
        std::unique_ptr<IVectorStorage> vector_storage_;
        LSMVecDBOptions db_options_;

    private:
        int randomLevel();
        float computeDistance(Span<const float> vectorA,
                              Span<const float> vectorB) const;
        void storeVectorWithStats(node_id_t id,
                                  const std::vector<float>& vec,
                                  node_id_t sectionKey);
        void readVectorWithStats(node_id_t id,
                                 std::vector<float>& vec);
        std::vector<SearchResult> searchLayer(const std::vector<float> &queryVector,
                                              node_id_t entryPointId,
                                              int efSearch,
                                              int layer);
        void linkNeighbors(node_id_t nodeId, const std::vector<node_id_t> &neighborIds, int layer);
        void linkNeighborsAsterDB(node_id_t nodeId, const std::vector<node_id_t> &neighborIds);

        std::vector<node_id_t> selectNeighbors(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsSimple(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic1(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic2(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        int m_;                      // Number of established connections
        int m_max_;                  // Maximum neighbors per layer
        int m_level_;                // Normalization factor for level generation
        float ef_construction_;      // Parameter for candidate selection

        std::unique_ptr<rocksdb::RocksGraph> db_;
        rocksdb::Options options_;
        
        std::ostream &out_file_;     // Output stream for logging

        std::unordered_map<int, Node> nodes_; // In-memory nodes for layers > 0

        std::random_device random_device_;
        std::mt19937 random_generator_;
        std::uniform_real_distribution<> uniform_distribution_;
        int max_layer_;
        node_id_t entry_point_ = k_invalid_node_id; // Entry point for HNSW graph

        std::unordered_set<node_id_t> deleted_ids_;

    };
} // namespace ROCKSDB_NAMESPACE
