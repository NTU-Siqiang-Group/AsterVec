#include "lsm_vec_index.h"
#include "disk_vector.h"
#include "distance.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>


uint8_t encode(int value)
{
    if (value < 0)
        return static_cast<uint8_t>(value + 128);
    else
        return static_cast<uint8_t>(value + 127);
}

int decode(uint8_t stored_value)
{
    if (stored_value < 128)
        return static_cast<int16_t>(stored_value - 128);
    else
        return static_cast<int16_t>(stored_value - 127);
}

float uniform( 
    float min, 
    float max) 
{
    if (min > max)
    {
        LOG(ERR) << "uniform input error: min > max";
        throw std::runtime_error("uniform input error: min > max");
    }

    float x = min + (max - min) * (float)rand() / (float)RAND_MAX;
    if (x < min || x > max)
    {
        LOG(ERR) << "uniform input error: generated value out of range";
        throw std::runtime_error("uniform input error: generated value out of range");
    }

    return x;
}

float gaussian(  // r.v. from Gaussian(mean, sigma)
    float mu,    // mean (location)
    float sigma) // stanard deviation (scale > 0)
{
    float PI = 3.141592654F;
    float FLOATZERO = 1e-6F;

    if (sigma <= 0.0f)
    {
        LOG(ERR) << "gaussian input error: sigma must be positive";
        throw std::runtime_error("gaussian input error: sigma must be positive");
    }

    float u1, u2;
    do
    {
        u1 = uniform(0.0f, 1.0f);
    } while (u1 < FLOATZERO);
    u2 = uniform(0.0f, 1.0f);

    float x = mu + sigma * sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);
    return x;
}

namespace lsm_vec
{
using namespace ROCKSDB_NAMESPACE;

    LSMVec::LSMVec(const std::string& db_path,
                   const LSMVecDBOptions& options,
                   std::ostream &outFile)
        : vector_dim_(options.dim),
          db_options_(options),
          m_(options.m),
          m_max_(options.m_max),
          m_level_(options.m_level),
          ef_construction_(options.ef_construction),
          out_file_(outFile),
          random_generator_(random_device_()),
          uniform_distribution_(0, 1),
          max_layer_(-1),
          entry_point_(-1)
    {
        stats.setEnabled(db_options_.enable_stats);

        if (db_options_.random_seed > 0) {
            random_generator_.seed(db_options_.random_seed);
        }

        if (db_options_.vector_file_path.empty()) {
            db_options_.vector_file_path = db_path + "/vector.log";
        }
        
        options_.create_if_missing = true;
        options_.db_paths.emplace_back(rocksdb::DbPath(db_path, db_options_.db_target_size));
        options_.statistics = rocksdb::CreateDBStatistics();

        db_ = std::make_unique<rocksdb::RocksGraph>(
            options_,
            EDGE_UPDATE_EAGER,
            ENCODING_TYPE_NONE,
            db_options_.reinit
        );

        if (db_options_.vector_storage_type == 1) {
            LOG(INFO) << "Using page-based vector storage layout";
            vector_storage_ = std::make_unique<PagedVectorStorage>(
                db_options_.vector_file_path,
                static_cast<size_t>(vector_dim_),
                db_options_.vec_file_capacity,
                db_options_.paged_max_cached_pages
            );
        } else {
            LOG(INFO) << "Using plain vector storage layout";
            vector_storage_ = std::make_unique<BasicVectorStorage>(
                db_options_.vector_file_path,
                static_cast<size_t>(vector_dim_),
                db_options_.vec_file_capacity
            );
        }
    }

    namespace {
    constexpr char kMetadataMagic[] = "LSMVMETA";
    constexpr uint32_t kMetadataVersion = 1;

    template <typename T>
    bool WriteValue(std::ostream& out, const T& value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        return static_cast<bool>(out);
    }

    template <typename T>
    bool ReadValue(std::istream& in, T* value)
    {
        in.read(reinterpret_cast<char*>(value), sizeof(T));
        return static_cast<bool>(in);
    }
    } // namespace

    Status LSMVec::SerializeMetadata(std::ostream& out) const
    {
        if (!out) {
            return Status::IOError("metadata output stream is not ready");
        }

        if (!out.write(kMetadataMagic, sizeof(kMetadataMagic) - 1)) {
            return Status::IOError("failed to write metadata magic");
        }
        if (!WriteValue(out, kMetadataVersion)) {
            return Status::IOError("failed to write metadata version");
        }

        uint64_t entryPoint = static_cast<uint64_t>(entry_point_);
        int32_t maxLayer = static_cast<int32_t>(max_layer_);
        uint64_t nodeCount = static_cast<uint64_t>(nodes_.size());
        uint64_t deletedCount = static_cast<uint64_t>(deleted_ids_.size());

        if (!WriteValue(out, entryPoint) ||
            !WriteValue(out, maxLayer) ||
            !WriteValue(out, nodeCount) ||
            !WriteValue(out, deletedCount)) {
            return Status::IOError("failed to write metadata header");
        }

        for (const auto& kv : nodes_) {
            node_id_t nodeId = kv.first;
            const Node& node = kv.second;
            uint64_t pointSize = static_cast<uint64_t>(node.point.size());
            uint64_t neighborLayers = static_cast<uint64_t>(node.neighbors.size());

            if (!WriteValue(out, nodeId) ||
                !WriteValue(out, pointSize)) {
                return Status::IOError("failed to write node metadata");
            }
            if (!node.point.empty()) {
                out.write(reinterpret_cast<const char*>(node.point.data()),
                          static_cast<std::streamsize>(node.point.size() * sizeof(float)));
                if (!out) {
                    return Status::IOError("failed to write node vector");
                }
            }

            if (!WriteValue(out, neighborLayers)) {
                return Status::IOError("failed to write neighbor layer count");
            }
            for (const auto& layerEntry : node.neighbors) {
                int32_t layer = static_cast<int32_t>(layerEntry.first);
                const auto& neighbors = layerEntry.second;
                uint64_t neighborCount = static_cast<uint64_t>(neighbors.size());
                if (!WriteValue(out, layer) || !WriteValue(out, neighborCount)) {
                    return Status::IOError("failed to write neighbors metadata");
                }
                for (node_id_t neighbor : neighbors) {
                    if (!WriteValue(out, neighbor)) {
                        return Status::IOError("failed to write neighbor id");
                    }
                }
            }
        }

        for (node_id_t id : deleted_ids_) {
            if (!WriteValue(out, id)) {
                return Status::IOError("failed to write deleted id");
            }
        }

        int32_t storageType = db_options_.vector_storage_type;
        if (!WriteValue(out, storageType)) {
            return Status::IOError("failed to write vector storage type");
        }
        if (storageType == 1) {
            auto* paged = dynamic_cast<PagedVectorStorage*>(vector_storage_.get());
            if (!paged) {
                return Status::IOError("paged vector storage not available");
            }
            if (!paged->serializeMetadata(out)) {
                return Status::IOError("failed to write paged storage metadata");
            }
        }

        return Status::OK();
    }

    Status LSMVec::DeserializeMetadata(std::istream& in)
    {
        if (!in) {
            return Status::IOError("metadata input stream is not ready");
        }

        char magic[sizeof(kMetadataMagic) - 1] = {};
        in.read(magic, sizeof(magic));
        if (!in || std::string(magic, sizeof(magic)) != kMetadataMagic) {
            return Status::IOError("invalid metadata magic");
        }

        uint32_t version = 0;
        if (!ReadValue(in, &version) || version < 1 || version > kMetadataVersion) {
            return Status::IOError("unsupported metadata version");
        }

        uint64_t entryPoint = 0;
        int32_t maxLayer = 0;
        uint64_t nodeCount = 0;
        uint64_t deletedCount = 0;
        if (!ReadValue(in, &entryPoint) ||
            !ReadValue(in, &maxLayer) ||
            !ReadValue(in, &nodeCount) ||
            !ReadValue(in, &deletedCount)) {
            return Status::IOError("failed to read metadata header");
        }

        nodes_.clear();
        nodes_.reserve(static_cast<size_t>(nodeCount));
        for (uint64_t i = 0; i < nodeCount; ++i) {
            node_id_t nodeId = 0;
            uint64_t pointSize = 0;
            if (!ReadValue(in, &nodeId) || !ReadValue(in, &pointSize)) {
                return Status::IOError("failed to read node metadata");
            }
            if (pointSize != static_cast<uint64_t>(vector_dim_)) {
                return Status::InvalidArgument("node vector dimension mismatch");
            }
            std::vector<float> point(static_cast<size_t>(pointSize));
            if (!point.empty()) {
                in.read(reinterpret_cast<char*>(point.data()),
                        static_cast<std::streamsize>(point.size() * sizeof(float)));
                if (!in) {
                    return Status::IOError("failed to read node vector");
                }
            }

            uint64_t neighborLayers = 0;
            if (!ReadValue(in, &neighborLayers)) {
                return Status::IOError("failed to read neighbor layer count");
            }
            Node node{nodeId, std::move(point), {}};
            for (uint64_t j = 0; j < neighborLayers; ++j) {
                int32_t layer = 0;
                uint64_t neighborCount = 0;
                if (!ReadValue(in, &layer) || !ReadValue(in, &neighborCount)) {
                    return Status::IOError("failed to read neighbor metadata");
                }
                std::vector<node_id_t> neighbors;
                neighbors.reserve(static_cast<size_t>(neighborCount));
                for (uint64_t k = 0; k < neighborCount; ++k) {
                    node_id_t neighbor = 0;
                    if (!ReadValue(in, &neighbor)) {
                        return Status::IOError("failed to read neighbor id");
                    }
                    neighbors.push_back(neighbor);
                }
                node.neighbors.emplace(layer, std::move(neighbors));
            }
            nodes_.emplace(nodeId, std::move(node));
        }

        deleted_ids_.clear();
        for (uint64_t i = 0; i < deletedCount; ++i) {
            node_id_t id = 0;
            if (!ReadValue(in, &id)) {
                return Status::IOError("failed to read deleted ids");
            }
            deleted_ids_.insert(id);
        }

        int32_t storageType = 0;
        if (!ReadValue(in, &storageType)) {
            return Status::IOError("failed to read vector storage type");
        }
        if (storageType != db_options_.vector_storage_type) {
            return Status::InvalidArgument("vector storage type mismatch");
        }
        if (storageType == 1) {
            auto* paged = dynamic_cast<PagedVectorStorage*>(vector_storage_.get());
            if (!paged) {
                return Status::IOError("paged vector storage not available");
            }
            if (!paged->deserializeMetadata(in)) {
                return Status::IOError("failed to load paged storage metadata");
            }
        }

        entry_point_ = static_cast<node_id_t>(entryPoint);
        max_layer_ = maxLayer;
        return Status::OK();
    }



    // Generates a random level for the node
    int LSMVec::randomLevel()
    {
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        double r = -log(distribution(random_generator_)) / log(1.0 * m_);

        // std::cout << "r: " << r << std::endl;
        return (int)r;
    }

    float LSMVec::computeDistance(Span<const float> vectorA,
                                  Span<const float> vectorB) const
    {
        if (vectorA.size() != vectorB.size())
        {
            throw std::invalid_argument("vector size mismatch");
        }

        return distance::ComputeDistance(
            db_options_.metric,
            vectorA.data(),
            vectorB.data(),
            vectorA.size());
    }

    void LSMVec::storeVectorWithStats(node_id_t id,
                                      const std::vector<float>& vec,
                                      node_id_t sectionKey)
    {
        auto timer = stats.startTimer();
        try {
            vector_storage_->storeVectorToDisk(id, vec, sectionKey);
        } catch (...) {
            stats.accumulateTime(timer, stats.vec_write_time);
            stats.addCount(1, stats.vec_write_count);
            throw;
        }
        stats.accumulateTime(timer, stats.vec_write_time);
        stats.addCount(1, stats.vec_write_count);
    }

    void LSMVec::readVectorWithStats(node_id_t id,
                                     std::vector<float>& vec)
    {
        auto timer = stats.startTimer();
        try {
            vector_storage_->readVectorFromDisk(id, vec);
        } catch (...) {
            stats.accumulateTime(timer, stats.vec_read_time);
            stats.addCount(1, stats.vec_read_count);
            throw;
        }
        stats.accumulateTime(timer, stats.vec_read_time);
        stats.addCount(1, stats.vec_read_count);
    }

    void LSMVec::insertNode(node_id_t nodeId, const std::vector<float> &vector)
    {
        deleted_ids_.erase(nodeId);
        bool vectorStored = false;  // Track whether we've stored this vector

        auto insert_timer = stats.startTimer();
        int highestLayer = randomLevel();

        if (highestLayer > 0)
        {
            Node newNode{nodeId, vector, {}};
            newNode.neighbors = std::unordered_map<int, std::vector<node_id_t>>();
            nodes_[nodeId] = newNode;
        }

        // ----- First node special case -----
        if (entry_point_ == k_invalid_node_id)
        {
            entry_point_ = nodeId;
            max_layer_   = highestLayer;

            linkNeighborsAsterDB(nodeId, {});

            // For the first node, we can just use nodeId as sectionKey
            node_id_t sectionKey = nodeId;
            storeVectorWithStats(nodeId, vector, sectionKey);
            vectorStored = true;

            stats.accumulateTime(insert_timer, stats.indexing_time);
            stats.addCount(1, stats.insert_count);
            if (insert_timer.active) {
                LOG(INFO) << "insert_node id=" << nodeId
                          << " layer=" << highestLayer
                          << " time_s=" << insert_timer.duration;
            }
            return;
        }

        node_id_t currentEntryPoint = entry_point_;

        auto extractIds = [](const std::vector<SearchResult>& results) {
            std::vector<node_id_t> ids;
            ids.reserve(results.size());
            for (const auto& result : results) {
                ids.push_back(result.id);
            }
            return ids;
        };

        // ----- Search down from top layer to (highestLayer+1) to choose entry -----
        for (int l = max_layer_; l > highestLayer; --l)
        {
            std::vector<SearchResult> closest =
                searchLayer(vector, currentEntryPoint, 1, l);
            if (!closest.empty())
            {
                currentEntryPoint = closest[0].id;
            }
        }

        // Initial guess for sectionKey: if we have upper layers, use currentEntryPoint,
        // otherwise fall back to global entry_point_.
        node_id_t sectionKey = (max_layer_ >= 1 ? currentEntryPoint : entry_point_);

        // ----- From min(max_layer_, highestLayer) ... down to 0 -----
        for (int l = std::min(max_layer_, highestLayer); l >= 0; --l)
        {
            std::vector<SearchResult> neighbors =
                searchLayer(vector, currentEntryPoint, ef_construction_, l);
            std::vector<node_id_t> neighborIds = extractIds(neighbors);
            std::vector<node_id_t> selectedNeighbors = 
                selectNeighbors(vector, neighborIds, m_, l);
                
            // std::vector<node_id_t> selectedNeighbors;
            // if (l > 0)
            //     selectedNeighbors = selectNeighbors(point, neighbors, M, l);
            // else
            //     selectedNeighbors = selectNeighbors(point, neighbors, Mmax, l);

            // If we are at level 1, refine sectionKey using closest neighbor at l=1.
            if (l == 1)
            {
                if (!neighbors.empty())
                    sectionKey = neighbors[0].id;
                else
                    sectionKey = currentEntryPoint;
            }

            // When we first reach level 0, store the vector using sectionKey
            if (l == 0 && !vectorStored)
            {
                storeVectorWithStats(nodeId, vector, sectionKey);
                vectorStored = true;
            }

            // Link neighbors as before
            if (l > 0)
            {
                linkNeighbors(nodeId, selectedNeighbors, l);
            }
            else // l == 0
            {
                linkNeighborsAsterDB(nodeId, selectedNeighbors);
            }

            // ---- Shrink connections ----
            if (l > 0)
            {
                for (int neighbor : selectedNeighbors)
                {
                    std::vector<node_id_t> eConn = nodes_[neighbor].neighbors[l];
                    if (eConn.size() > static_cast<size_t>(m_max_))
                    {
                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(nodes_[neighbor].point, eConn, m_max_, l);
                        nodes_[neighbor].neighbors[l] = std::move(eNewConn);
                    }
                }
            }
            else // l == 0
            {
                for (node_id_t neighbor : selectedNeighbors)
                {
                    rocksdb::Edges edges;
                    db_->GetAllEdges(neighbor, &edges);

                    if (edges.num_edges_out > static_cast<uint32_t>(m_max_))
                    {
                        std::vector<node_id_t> eConns;
                        eConns.reserve(edges.num_edges_out);
                        for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                        {
                            eConns.push_back(edges.nxts_out[i].nxt);
                        }

                        std::vector<float> neighborVector;
                        readVectorWithStats(neighbor, neighborVector);

                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(neighborVector, eConns, m_max_, l);

                        for (auto node : eConns)
                        {
                            if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                            {
                                db_->DeleteEdge(neighbor, node);
                                db_->DeleteEdge(node, neighbor);
                            }
                        }
                    }
                    rocksdb::free_edges(&edges);
                }
            }

            if (!neighbors.empty())
            {
                currentEntryPoint = neighbors[0].id;
            }
        }

        if (highestLayer > max_layer_)
        {
            entry_point_ = nodeId;
            max_layer_   = highestLayer;
            LOG(INFO) << "Updated entry point to node " << nodeId
                      << " at layer " << highestLayer;
        }

        // Safety: in principle vectorStored must be true if we reached here.
        if (!vectorStored)
        {
            storeVectorWithStats(nodeId, vector, sectionKey);
        }

        stats.accumulateTime(insert_timer, stats.indexing_time);
        stats.addCount(1, stats.insert_count);
    }

    Status LSMVec::deleteNode(node_id_t id)
    {
        deleted_ids_.insert(id);
        try {
            vector_storage_->deleteVector(id);
        } catch (const std::exception& ex) {
            return Status::IOError(ex.what());
        }

        for (auto& kv : nodes_) {
            auto& neighbor_map = kv.second.neighbors;
            for (auto& layer_entry : neighbor_map) {
                auto& neighbors = layer_entry.second;
                neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), id), neighbors.end());
            }
        }

        nodes_.erase(id);

        rocksdb::Edges edges;
        db_->GetAllEdges(id, &edges);
        for (uint32_t i = 0; i < edges.num_edges_out; ++i) {
            node_id_t neighborId = static_cast<node_id_t>(edges.nxts_out[i].nxt);
            db_->DeleteEdge(id, neighborId);
            db_->DeleteEdge(neighborId, id);
        }
        rocksdb::free_edges(&edges);

        return Status::OK();
    }

    Status LSMVec::updateNode(node_id_t id, const std::vector<float>& vec)
    {
        Status delete_status = deleteNode(id);
        if (!delete_status.ok()) {
            return delete_status;
        }

        try {
            insertNode(id, vec);
        } catch (const std::exception& ex) {
            return Status::IOError(ex.what());
        }

        return Status::OK();
    }

    Status LSMVec::getNodeVector(node_id_t id, std::vector<float>* out)
    {
        if (!out) {
            return Status::InvalidArgument("output vector must not be null");
        }
        if (deleted_ids_.count(id) > 0) {
            return Status::NotFound("vector deleted");
        }
        if (!vector_storage_->exists(id)) {
            return Status::NotFound("vector deleted");
        }

        try {
            readVectorWithStats(id, *out);
        } catch (const std::exception& ex) {
            return Status::IOError(ex.what());
        }

        return Status::OK();
    }

    std::vector<SearchResult> LSMVec::knnSearchK(const std::vector<float>& query, int k, int ef_search)
    {
        if (entry_point_ == k_invalid_node_id || k <= 0) {
            return {};
        }

        auto search_timer = stats.startTimer();
        node_id_t currentEntryPoint = entry_point_;
        for (int l = max_layer_; l >= 1; --l) {
            std::vector<SearchResult> nearest =
                searchLayer(query, currentEntryPoint, ef_search, l);
            if (!nearest.empty()) {
                currentEntryPoint = nearest[0].id;
            }
        }

        int ef = std::max(ef_search, k);
        std::vector<SearchResult> neighbors =
            searchLayer(query, currentEntryPoint, ef, 0);
        if (neighbors.empty()) {
            return neighbors;
        }

        std::vector<SearchResult> filtered;
        filtered.reserve(static_cast<size_t>(k));
        for (const auto& result : neighbors) {
            if (deleted_ids_.count(result.id) > 0) {
                continue;
            }
            filtered.push_back(result);
            if (static_cast<int>(filtered.size()) >= k) {
                break;
            }
        }

        stats.accumulateTime(search_timer, stats.search_time);
        stats.addCount(1, stats.search_count);
        if (search_timer.active) {
            DLOG(DEBUG) << "knn_search_k k=" << k
                        << " time_s=" << search_timer.duration;
        }
        return filtered;
    }

    // Links neighbors for upper layers stored in memory
    void LSMVec::linkNeighbors(node_id_t nodeId, const std::vector<node_id_t> &neighborIds, int layer)
    {
        for (node_id_t neighborId : neighborIds)
        {
            nodes_[neighborId].neighbors[layer].push_back(nodeId);
            nodes_[nodeId].neighbors[layer].push_back(neighborId);
        }
    }

    void LSMVec::linkNeighborsAsterDB(node_id_t nodeId, const std::vector<node_id_t> &neighborIds)
    {
        db_->AddVertex(nodeId);

        for (node_id_t neighborId : neighborIds)
        {
            db_->AddEdge(nodeId, neighborId);
            db_->AddEdge(neighborId, nodeId);
        }
    }

    std::vector<node_id_t> LSMVec::selectNeighbors(
        const std::vector<float> &vector,
        const std::vector<node_id_t> &candidateIds,
        int maxNeighbors,
        int layer)
    {
        if (!use_heuristic_neighbor_selection_) {
            return selectNeighborsSimple(vector, candidateIds, maxNeighbors, layer);
        } else {
            return selectNeighborsHeuristic2(vector, candidateIds, maxNeighbors, layer);
        }
    }

    // Selects neighbors based on distance and pruning logic
    std::vector<node_id_t> LSMVec::selectNeighborsSimple(const std::vector<float> &vector, const std::vector<node_id_t> &candidateIds, int maxNeighbors, int layer)
    {
        if (candidateIds.size() <= static_cast<size_t>(maxNeighbors))
        {
            return candidateIds;
        }
        else
        {
            std::priority_queue<std::pair<float, node_id_t>> topCandidates;
            for (node_id_t candidateId : candidateIds)
            {
                float dist = 0.0;
                if (layer > 0)
                {
                    const auto& candidateVec = nodes_[candidateId].point;
                    dist = computeDistance(Span<const float>(vector),
                                           Span<const float>(candidateVec));
                }
                else if (layer == 0)
                {
                    std::vector<float> candidateVector;
                    readVectorWithStats(candidateId, candidateVector);
                    dist = computeDistance(Span<const float>(vector),
                                           Span<const float>(candidateVector));
                }

                topCandidates.emplace(dist, candidateId);
                if (topCandidates.size() > static_cast<size_t>(maxNeighbors))
                {
                    topCandidates.pop();
                }
            }

            std::vector<node_id_t> selectedNeighbors;
            while (!topCandidates.empty())
            {
                selectedNeighbors.push_back(topCandidates.top().second);
                topCandidates.pop();
            }
            return selectedNeighbors;
        }
    }

    std::vector<node_id_t> LSMVec::selectNeighborsHeuristic1(
        const std::vector<float> &vector,
        const std::vector<node_id_t> &candidateIds,
        int maxNeighbors,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidateIds.size() <= static_cast<size_t>(maxNeighbors)) {
            return candidateIds;
        }

        struct CandidateInfo {
            node_id_t   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidateIds.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes_[id].point directly.
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t nodeId) -> const std::vector<float>& {
            if (layer > 0) {
                return nodes_[nodeId].point;
            } else {
                auto it = vecCache.find(nodeId);
                if (it != vecCache.end()) return it->second;

                std::vector<float> v;
                readVectorWithStats(nodeId, v);

                auto res = vecCache.emplace(nodeId, std::move(v));
                return res.first->second;
            }
        };

        // 1) Precompute distances query -> candidate
        for (node_id_t candidateId : candidateIds)
        {
            const auto& candVec = getVector(candidateId);
            float d = computeDistance(Span<const float>(vector),
                                      Span<const float>(candVec));
            candInfos.push_back(CandidateInfo{candidateId, d});
        }

        // 2) Sort by distance to query (ascending)
        std::sort(
            candInfos.begin(),
            candInfos.end(),
            [](const CandidateInfo& a, const CandidateInfo& b) {
                return a.distToQuery < b.distToQuery;
            }
        );

        // 3) Heuristic (diversified neighbor selection):
        std::vector<node_id_t> selected;
        selected.reserve(maxNeighbors);
        std::vector<node_id_t> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(maxNeighbors))
                break;

            node_id_t candidateId = cand.id;
            float distToQuery = cand.distToQuery;

            bool good = true;
            for (node_id_t selectedId : selected)
            {
                const auto& v1 = getVector(selectedId);
                const auto& v2 = getVector(candidateId);
                float currentDist = computeDistance(Span<const float>(v1),
                                                    Span<const float>(v2));

                if (currentDist < distToQuery)
                {
                    good = false;
                    break;
                }
            }

            if (good)
                selected.push_back(candidateId);
            else
                rejected.push_back(candidateId);
        }

        // 4) If we still have fewer than maxNeighbors neighbors, fill from rejected,
        //    preserving the order by distToQuery (because 'rejected' was
        //    filled while scanning candInfos in sorted order).
        for (node_id_t rejectedId : rejected)
        {
            if (selected.size() >= static_cast<size_t>(maxNeighbors))
                break;
            selected.push_back(rejectedId);
        }

        return selected;
    }

    std::vector<node_id_t> LSMVec::selectNeighborsHeuristic2(
        const std::vector<float> &vector,
        const std::vector<node_id_t> &candidateIds,
        int maxNeighbors,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidateIds.size() <= static_cast<size_t>(maxNeighbors)) {
            return candidateIds;
        }

        struct CandidateInfo {
            node_id_t   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidateIds.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes_[id].point directly.
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t nodeId) -> const std::vector<float>& {
            if (layer > 0) {
                return nodes_[nodeId].point;
            } else {
                auto it = vecCache.find(nodeId);
                if (it != vecCache.end()) return it->second;

                std::vector<float> v;
                readVectorWithStats(nodeId, v);

                auto res = vecCache.emplace(nodeId, std::move(v));
                return res.first->second;
            }
        };

        // 1) Precompute distances query -> candidate
        for (node_id_t candidateId : candidateIds)
        {
            const auto& candVec = getVector(candidateId);
            float d = computeDistance(Span<const float>(vector),
                                      Span<const float>(candVec));
            candInfos.push_back(CandidateInfo{candidateId, d});
        }

        // 2) Sort by distance to query (ascending)
        std::sort(
            candInfos.begin(),
            candInfos.end(),
            [](const CandidateInfo& a, const CandidateInfo& b) {
                return a.distToQuery < b.distToQuery;
            }
        );

        // 3) Heuristic (diversified neighbor selection):
        std::vector<node_id_t> selected;
        selected.reserve(maxNeighbors);
        std::vector<node_id_t> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(maxNeighbors))
                break;

            node_id_t candidateId = cand.id;
            float distToQuery = cand.distToQuery;

            bool good = true;
            for (node_id_t selectedId : selected)
            {
                const auto& v1 = getVector(selectedId);
                const auto& v2 = getVector(candidateId);
                float currentDist = computeDistance(Span<const float>(v1),
                                                    Span<const float>(v2));

                if (currentDist < distToQuery)
                {
                    good = false;
                    break;
                }
            }

            if (good)
                selected.push_back(candidateId);
            else
                rejected.push_back(candidateId);
        }

        return selected;
    }

    std::vector<SearchResult> LSMVec::searchLayer(const std::vector<float>& queryVector,
                                                  node_id_t entryPointId,
                                                  int efSearch,
                                                  int layer)
    {
        if (layer < 0) {
            LOG(ERR) << "Invalid layer for search";
            return {};
        }
        if (efSearch <= 0) {
            return {};
        }

        // Visited set
        std::unordered_set<node_id_t> visited;
        visited.reserve(static_cast<std::size_t>(efSearch) * 4);

        // Candidates: max-heap by (-distance) => smallest distance comes first
        using Cand = std::pair<float, node_id_t>;
        std::priority_queue<Cand> candidates;

        // W: max-heap by (distance) => farthest among current best is on top
        std::priority_queue<Cand> nearest;

        auto getDistance = [&](node_id_t nodeId) -> float {
            if (layer > 0) {
                // Upper layers: vectors are in-memory
                const auto& point = nodes_.at(nodeId).point;
                return computeDistance(Span<const float>(queryVector),
                                       Span<const float>(point));
            } else {
                // Level 0: vectors are on disk
                std::vector<float> v;
                readVectorWithStats(nodeId, v);
                return computeDistance(Span<const float>(queryVector),
                                       Span<const float>(v));
            }
        };

        // Initialize with entry point
        float distToEntry = getDistance(entryPointId);
        visited.insert(entryPointId);
        candidates.emplace(-distToEntry, entryPointId);
        nearest.emplace(distToEntry, entryPointId);

        while (!candidates.empty()) {
            node_id_t currentId = candidates.top().second;
            float currentDist = -candidates.top().first;
            candidates.pop();

            // Early termination: if closest candidate is worse than the worst in W, stop
            if (!nearest.empty() && currentDist > nearest.top().first) {
                break;
            }

            if (layer > 0) {
                // Upper layers: adjacency is in memory.
                const auto nodeIt = nodes_.find(currentId);
                if (nodeIt == nodes_.end()) {
                    continue; // Defensive: should not happen
                }

                const auto& neighborMap = nodeIt->second.neighbors;
                auto it = neighborMap.find(layer);
                if (it == neighborMap.end()) {
                    continue; // No adjacency list at this layer (do not create it)
                }

                const auto& neighborIds = it->second;
                for (node_id_t neighborId : neighborIds) {
                    if (visited.insert(neighborId).second) {
                        const auto& point = nodes_.at(neighborId).point;
                        float d = computeDistance(Span<const float>(queryVector),
                                                  Span<const float>(point));
                        if (static_cast<int>(nearest.size()) < efSearch || d < nearest.top().first) {
                            candidates.emplace(-d, neighborId);
                            nearest.emplace(d, neighborId);
                            if (static_cast<int>(nearest.size()) > efSearch) {
                                nearest.pop();
                            }
                        }
                    }
                }
            } else {
                // Level 0: adjacency is stored in RocksGraph.
                rocksdb::Edges edges;
                db_->GetAllEdges(currentId, &edges);

                for (uint32_t i = 0; i < edges.num_edges_out; ++i) {
                    node_id_t neighborId = static_cast<node_id_t>(edges.nxts_out[i].nxt);

                    if (visited.insert(neighborId).second) {
                        std::vector<float> neighborVec;
                        readVectorWithStats(neighborId, neighborVec);
                        float d = computeDistance(Span<const float>(queryVector),
                                                  Span<const float>(neighborVec));
                        if (static_cast<int>(nearest.size()) < efSearch || d < nearest.top().first) {
                            candidates.emplace(-d, neighborId);
                            nearest.emplace(d, neighborId);
                            if (static_cast<int>(nearest.size()) > efSearch) {
                                nearest.pop();
                            }
                        }
                    }
                }
                rocksdb::free_edges(&edges);
            }
        }

        // Extract results from nearest (sorted ascending by distance)
        std::vector<std::pair<float, node_id_t>> temp;
        temp.reserve(nearest.size());
        while (!nearest.empty()) {
            temp.push_back(nearest.top());
            nearest.pop();
        }

        std::sort(temp.begin(), temp.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<SearchResult> result;
        result.reserve(temp.size());
        for (const auto& p : temp) {
            result.push_back({p.second, p.first});
        }
        return result;
    }


    // std::vector<node_id_t> LSMVec::searchLayer(const std::vector<float> &queryPoint, node_id_t entryPoint, int ef, int layer)
    // {
    //     // set of visited elements
    //     std::unordered_set<node_id_t> visited;

    //     // set of candidates
    //     std::priority_queue<std::pair<float, node_id_t>> candidates; // C: set of candidates

    //     // dynamic list of found nearest neighbors
    //     std::priority_queue<std::pair<float, node_id_t>> nearestNeighbors; // W: dynamic list of found nearest neighbors

    //     if (layer > 0)
    //     {
    //         // initialize the search
    //         float distToEP = euclideanDistance(queryPoint, nodes[entryPoint].point);
    //         visited.insert(entryPoint); // v ← ep
    //         candidates.emplace(-distToEP, entryPoint);
    //         nearestNeighbors.emplace(distToEP, entryPoint);

    //         while (!candidates.empty())
    //         {
    //             // get the nearest candidate, extract nearest element from C
    //             node_id_t current = candidates.top().second;
    //             float currentDist = -candidates.top().first;
    //             candidates.pop();

    //             // check if the current candidate is closer than the farthest neighbor
    //             // get furthest element from W to q
    //             if (currentDist > nearestNeighbors.top().first)
    //             {
    //                 break;
    //             }
    //             for (node_id_t neighbor : nodes[current].neighbors[layer])
    //             {
    //                 if (visited.find(neighbor) == visited.end())
    //                 {
    //                     visited.insert(neighbor);
    //                     float dist = euclideanDistance(queryPoint, nodes[neighbor].point);
    //                     if (nearestNeighbors.size() < ef || dist < nearestNeighbors.top().first)
    //                     {
    //                         candidates.emplace(-dist, neighbor);
    //                         nearestNeighbors.emplace(dist, neighbor);
    //                         if (nearestNeighbors.size() > ef)
    //                         {
    //                             nearestNeighbors.pop();
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         std::vector<std::pair<float, node_id_t>> temp;

            
    //         while (!nearestNeighbors.empty())
    //         {
    //             temp.push_back(nearestNeighbors.top());
    //             nearestNeighbors.pop();
    //         }

    //         std::sort(temp.begin(), temp.end(), [](const std::pair<float, node_id_t> &a, const std::pair<float, node_id_t> &b)
    //                   {
    //                       return a.first < b.first;
    //                   });

    //         std::vector<node_id_t> result;
    //         for (const auto &pair : temp)
    //         {
    //             result.push_back(pair.second);
    //         }
    //         return result;
    //     }
    //     else if (layer == 0)
    //     {
    //         // initialize the search
    //         // float distToEP = euclideanDistance(queryPoint, nodes[entryPoint].point);
    //         std::vector<float> entryPointVector;

    //         vectorStorage->readVectorFromDisk(entryPoint, entryPointVector);

    //         float distToEP = euclideanDistance(queryPoint, entryPointVector);
    //         visited.insert(entryPoint); // v ← ep
    //         candidates.emplace(-distToEP, entryPoint);
    //         nearestNeighbors.emplace(distToEP, entryPoint);

    //         while (!candidates.empty())
    //         {
    //             // extract nearest element from C to q
    //             // get furthest element from W to q
    //             node_id_t current = candidates.top().second;
    //             float currentDist = -candidates.top().first;
    //             candidates.pop();

    //             // check if the current candidate is closer than the farthest neighbor
    //             // get furthest element from W to q
    //             if (currentDist > nearestNeighbors.top().first)
    //             {
    //                 break;
    //             }

    //             rocksdb::Edges edges;
    //             db_->GetAllEdges(current, &edges);

    //             std::vector<int> neighbors;
    //             for (uint32_t i = 0; i < edges.num_edges_out; ++i)
    //             {
    //                 neighbors.push_back(edges.nxts_out[i].nxt);
    //             }

    //             for (node_id_t neighbor : neighbors)
    //             {
    //                 if (visited.find(neighbor) == visited.end())
    //                 {
    //                     visited.insert(neighbor);
    //                     // float dist = euclideanDistance(queryPoint, nodes[neighbor].point);
    //                     std::vector<float> neighborVector;
    //                     vectorStorage->readVectorFromDisk(neighbor, neighborVector);

    //                     float dist = euclideanDistance(queryPoint, neighborVector);
    //                     if (nearestNeighbors.size() < ef || dist < nearestNeighbors.top().first)
    //                     {
    //                         candidates.emplace(-dist, neighbor);
    //                         nearestNeighbors.emplace(dist, neighbor);
    //                         if (nearestNeighbors.size() > ef)
    //                         {
    //                             nearestNeighbors.pop(); // remove furthest element from W to q
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         std::vector<std::pair<float, node_id_t>> temp;

    //         while (!nearestNeighbors.empty())
    //         {
    //             temp.push_back(nearestNeighbors.top());
    //             nearestNeighbors.pop();
    //         }

    //         std::sort(temp.begin(), temp.end(), [](const std::pair<float, node_id_t> &a, const std::pair<float, node_id_t> &b)
    //                   {
    //                       return a.first < b.first; // 比较距离
    //                   });

    //         std::vector<node_id_t> result;
    //         for (const auto &pair : temp)
    //         {
    //             result.push_back(pair.second);
    //         }
    //         return result;
    //     }
    //     else
    //     {
    //         // error
    //         std::cerr << "Error: Invalid layer for search." << std::endl;
    //     }
    // }

    // Performs a greedy search to find the closest neighbor at a specific layer
    node_id_t LSMVec::knnSearch(const std::vector<float> &queryVector)
    {
        auto search_timer = stats.startTimer();
        // W ← ∅ set for the current nearest elements
        std::vector<SearchResult> nearestNeighbors; // W: dynamic list of found nearest neighbors

        // ep ← get enter point for hnsw
        node_id_t currentEntryPoint = entry_point_;
        int currentLayer = max_layer_;

        for (int l = max_layer_; l >= 1; --l)
        {
            std::vector<SearchResult> nearestNeighbors =
                searchLayer(queryVector, currentEntryPoint, 30, l);
            currentEntryPoint = nearestNeighbors[0].id;
        }
        nearestNeighbors = searchLayer(queryVector, currentEntryPoint, 30, 0);

        stats.accumulateTime(search_timer, stats.search_time);
        stats.addCount(1, stats.search_count);
        return nearestNeighbors[0].id;
    }

    void LSMVec::printState() const
    {
        // We do not print layer 0 by request.
        if (max_layer_ <= 0) {
            LOG(INFO) << "HNSW state: max_layer=" << max_layer_
                      << " (no upper layers to report)";
            return;
        }

        // Count how many nodes have adjacency info at each upper layer.
        // Note: this counts "nodes that currently have neighbor entries at layer l",
        // which is derivable from existing in-memory structures without extra metadata.
        std::vector<std::size_t> layerNodeCounts(static_cast<std::size_t>(max_layer_ + 1), 0);

        // To avoid double counting, we track per-layer seen node IDs.
        std::vector<std::unordered_set<node_id_t>> seen(static_cast<std::size_t>(max_layer_ + 1));

        for (const auto& kv : nodes_) {
            node_id_t nodeId = kv.first;
            const Node& node = kv.second;

            for (const auto& kv2 : node.neighbors) {
                // The key type should be "layer index".
                // If your neighbors map key is int, this is already int.
                // If it's not int, cast safely.
                int layer = static_cast<int>(kv2.first);

                if (layer <= 0) continue;          // skip layer 0 (and any invalid)
                if (layer > max_layer_) continue;    // defensive

                // Option A (default): count node if it has the layer key at all.
                // Option B: count only if the adjacency list is non-empty:
                // if (kv2.second.empty()) continue;

                if (seen[static_cast<std::size_t>(layer)].insert(nodeId).second) {
                    layerNodeCounts[static_cast<std::size_t>(layer)]++;
                }
            }
        }

        std::ostringstream oss;
        oss << "HNSW state:\n";
        oss << "  max_layer: " << max_layer_ << "\n";
        oss << "  upper_layers: [1.." << max_layer_ << "]\n";

        // Print top-down for readability
        for (int l = max_layer_; l >= 1; --l) {
            oss << "  layer " << l << ": "
                << layerNodeCounts[static_cast<std::size_t>(l)]
                << " nodes\n";
        }
        LOG(INFO) << oss.str();
    }

    void LSMVec::printStatistics() const
    {
        auto cache_stats = vector_storage_->getPageCacheStats();
        stats.page_cache_hits = cache_stats.hits;
        stats.page_cache_misses = cache_stats.misses;

        std::ostringstream oss;
        stats.print(oss);
        LOG(INFO) << oss.str();
    }

    void LSMVec::close()
    {
        db_.reset();
    }

    // void LSMVec::printStatistics() const
    // {
    //     std::cout << "Indexing Time: " << indexingTime << " seconds" << std::endl;

    //     std::cout << "-------graph part------" << std::endl;
    //     std::cout << "Total Aster I/O Time: " << ioTime << " seconds" << std::endl;
    //     std::cout << "Read Operations: " << readIOCount << ", Time: " << readIOTime << " seconds" << std::endl;
    //     std::cout << "Write Node Operations: " << writenodeIOCount << ", Time: " << writenodeIOTime << " seconds" << std::endl;
    //     std::cout << "Add Edge Operations: " << addedgeIOCount << ", Time: " << addedgeIOTime << " seconds" << std::endl;
    //     std::cout << "Delete Edge Operations: " << deleteedgeIOCount << ", Time: " << deleteedgeIOTime << " seconds" << std::endl;
    //     std::cout << "-------vector part------" << std::endl;
    //     std::cout << "Total Vector I/O Time: " << vecreadtime + vecwritetime << " seconds" << std::endl;
    //     std::cout << "Vector Read Operations: " << vecreadcount << ", Time: " << vecreadtime << " seconds" << std::endl;
    //     std::cout << "Vector Write Operations: " << vecwritecount << ", Time: " << vecwritetime << " seconds" << std::endl;
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
} // namespace ROCKSDB_NAMESPACE
