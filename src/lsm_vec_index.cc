#include "lsm_vec_index.h"
#include "disk_vector.h"
#include "distance.h"
#include "metadata_store.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdio>


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
        enable_batch_read_ = db_options_.enable_batch_read;
        if (db_options_.edge_cache_size > 0) {
            edge_cache_ = std::make_unique<EdgeLRUCache>(db_options_.edge_cache_size);
            LOG(INFO) << "Edge LRU cache: " << db_options_.edge_cache_size << " entries";
        }

        if (db_options_.random_seed > 0) {
            random_generator_.seed(db_options_.random_seed);
        }

        if (db_options_.vector_file_path.empty()) {
            db_options_.vector_file_path = db_path + "/vector.log";
        }
        
        options_.create_if_missing = true;
        options_.db_paths.emplace_back(rocksdb::DbPath(db_path, db_options_.db_target_size));

        // RocksDB tuning: block cache, compaction parallelism, no compression
        {
            auto table_options =
                options_.table_factory->GetOptions<rocksdb::BlockBasedTableOptions>();
            if (table_options) {
                table_options->block_cache = rocksdb::NewLRUCache(32 * 1024 * 1024);
            }
            options_.max_background_jobs = 8;
            options_.max_background_compactions = 4;
            options_.max_subcompactions = 4;
            options_.compression = rocksdb::kNoCompression;
        }

        // Aster minimal_mode=true disables MorrisCounter degree-tracking and
        // in-neighbor maintenance. LSM-Vec doesn't use either, and minimal_mode
        // also makes the graph tolerate u64 ids with bit 63 set (our update_id
        // namespace) without the counter array trying to allocate ~exabytes.
        // Requires Aster's feature/minimal-mode branch.
        db_ = std::make_unique<rocksdb::RocksGraph>(
            options_,
            EDGE_UPDATE_EAGER,
            ENCODING_TYPE_NONE,
            db_options_.reinit,
            db_path,
            /*minimal_mode=*/true
        );

        nodes_.reserve(10000);

        // Reinit: remove stale vector storage files before construction
        if (db_options_.reinit) {
            std::remove(db_options_.vector_file_path.c_str());
            std::remove((db_options_.vector_file_path + ".deleted").c_str());
        }

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

        // Adaptive section layer: find smallest k such that M^k >= vectorsPerPage
        {
            size_t vpp = 1;
            if (auto* paged = dynamic_cast<PagedVectorStorage*>(vector_storage_.get()))
                vpp = paged->vectorsPerPage();

            section_layer_ = 1;
            size_t coverage = static_cast<size_t>(m_);
            while (coverage < vpp && section_layer_ < 10) {
                ++section_layer_;
                coverage *= static_cast<size_t>(m_);
            }
            LOG(INFO) << "Adaptive section layer: " << section_layer_
                      << " (vectorsPerPage=" << vpp << ", M=" << m_ << ")";
        }
    }

    namespace {
    constexpr char kMetadataMagic[] = "LSMVMETA";
    // v1: original metadata-filtering era format (entry_point, layers, nodes_, deleted_ids)
    // v2: + update sparse map + next_update_internal_id_ counter (C7, robust delete)
    constexpr uint32_t kMetadataVersion = 2;

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
        uint64_t deletedCount = static_cast<uint64_t>(tombstoned_internal_ids_.size());

        if (!WriteValue(out, entryPoint) ||
            !WriteValue(out, maxLayer) ||
            !WriteValue(out, nodeCount) ||
            !WriteValue(out, deletedCount)) {
            return Status::IOError("failed to write metadata header");
        }

        std::vector<float> serialize_dequant_scratch;
        for (const auto& kv : nodes_) {
            node_id_t nodeId = kv.first;
            const Node& node = kv.second;
            // 20260516_upper_layer_sq8: on-disk format is float32; dequantize
            // SQ8-resident nodes back to float for write. Format unchanged
            // across the SQ8 switch.
            const float* point_data = nullptr;
            size_t point_size = 0;
            if (!node.point.empty()) {
                point_data = node.point.data();
                point_size = node.point.size();
            } else if (!node.q8_data.empty()) {
                dequantizeNodePoint(node, serialize_dequant_scratch);
                point_data = serialize_dequant_scratch.data();
                point_size = serialize_dequant_scratch.size();
            }
            uint64_t pointSize = static_cast<uint64_t>(point_size);
            uint64_t neighborLayers = static_cast<uint64_t>(node.neighbors.size());

            if (!WriteValue(out, nodeId) ||
                !WriteValue(out, pointSize)) {
                return Status::IOError("failed to write node metadata");
            }
            if (point_data != nullptr && point_size > 0) {
                out.write(reinterpret_cast<const char*>(point_data),
                          static_cast<std::streamsize>(point_size * sizeof(float)));
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

        for (node_id_t id : tombstoned_internal_ids_) {
            if (!WriteValue(out, id)) {
                return Status::IOError("failed to write deleted id");
            }
        }

        // v2: update sparse forward map + monotonic counter.
        // Reverse map and Bloom filter are reconstructed on Open from this map.
        uint64_t updatedCount = static_cast<uint64_t>(updated_real_to_internal_.size());
        uint64_t nextUpdateId = static_cast<uint64_t>(next_update_internal_id_);
        if (!WriteValue(out, updatedCount) || !WriteValue(out, nextUpdateId)) {
            return Status::IOError("failed to write update map header");
        }
        for (const auto& kv : updated_real_to_internal_) {
            uint64_t r = static_cast<uint64_t>(kv.first);
            uint64_t i = static_cast<uint64_t>(kv.second);
            if (!WriteValue(out, r) || !WriteValue(out, i)) {
                return Status::IOError("failed to write update map entry");
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
            Node node;
            node.id = nodeId;
            node.point = std::move(point);
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
            // 20260516_upper_layer_sq8: every node in nodes_ is an upper-layer
            // node by construction (only highestLayer > 0 enters the map).
            // Re-quantize unconditionally so the in-memory form matches what
            // insertNode would have produced.
            if (!node.point.empty()) {
                quantizeNodePoint(node);
            }
            nodes_.emplace(nodeId, std::move(node));
        }

        tombstoned_internal_ids_.clear();
        for (uint64_t i = 0; i < deletedCount; ++i) {
            node_id_t id = 0;
            if (!ReadValue(in, &id)) {
                return Status::IOError("failed to read deleted ids");
            }
            tombstoned_internal_ids_.insert(id);
        }

        // v2: read update sparse map + counter, then rebuild reverse map and Bloom.
        // v1 files don't have this block — leave structures empty.
        updated_real_to_internal_.clear();
        updated_internal_to_real_.clear();
        next_update_internal_id_ = kFirstUpdateId;
        if (version >= 2) {
            uint64_t updatedCount = 0;
            uint64_t nextUpdateId = static_cast<uint64_t>(kFirstUpdateId);
            if (!ReadValue(in, &updatedCount) || !ReadValue(in, &nextUpdateId)) {
                return Status::IOError("failed to read update map header");
            }
            next_update_internal_id_ = static_cast<internal_id_t>(nextUpdateId);
            updated_real_to_internal_.reserve(static_cast<size_t>(updatedCount));
            for (uint64_t i = 0; i < updatedCount; ++i) {
                uint64_t r = 0, inter = 0;
                if (!ReadValue(in, &r) || !ReadValue(in, &inter)) {
                    return Status::IOError("failed to read update map entry");
                }
                updated_real_to_internal_[static_cast<real_id_t>(r)] =
                    static_cast<internal_id_t>(inter);
                updated_internal_to_real_[static_cast<internal_id_t>(inter)] =
                    static_cast<real_id_t>(r);
            }
        }
        // Rebuild Bloom filter from forward map. Capacity sized to current
        // entry count (with a 512 floor to match the constructor default).
        const std::size_t bloom_cap =
            std::max<std::size_t>(updated_real_to_internal_.size(), 512);
        update_bloom_.reset(bloom_cap, 0.01);
        for (const auto& kv : updated_real_to_internal_) {
            update_bloom_.add(kv.first);
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



    // 20260516_upper_layer_sq8: scalar quantize the float point to a (min, max,
    // uint8[dim]) form. Destructive — writes q8_data/q_min/q_max and clears
    // point. Idempotent: no-op if point is already empty.
    void LSMVec::quantizeNodePoint(Node& n)
    {
        if (n.point.empty()) return;
        const size_t d = n.point.size();
        float lo = n.point[0], hi = n.point[0];
        for (size_t i = 1; i < d; ++i) {
            if (n.point[i] < lo) lo = n.point[i];
            if (n.point[i] > hi) hi = n.point[i];
        }
        float range = hi - lo;
        if (range < 1e-10f) range = 1e-10f;
        n.q8_data.resize(d);
        for (size_t i = 0; i < d; ++i) {
            float norm = (n.point[i] - lo) / range;
            int q = static_cast<int>(norm * 255.0f + 0.5f);
            if (q < 0) q = 0; else if (q > 255) q = 255;
            n.q8_data[i] = static_cast<uint8_t>(q);
        }
        n.q_min = lo;
        n.q_max = hi;
        std::vector<float>{}.swap(n.point);   // free the float buffer
    }

    bool LSMVec::dequantizeNodePoint(const Node& n, std::vector<float>& out) const
    {
        if (n.q8_data.empty()) return false;
        const size_t d = n.q8_data.size();
        out.resize(d);
        const float range = n.q_max - n.q_min;
        for (size_t i = 0; i < d; ++i) {
            out[i] = n.q_min + range * (static_cast<float>(n.q8_data[i]) / 255.0f);
        }
        return true;
    }

    Span<const float> LSMVec::nodePointView(const Node& n,
                                              std::vector<float>& scratch) const
    {
        if (!n.point.empty()) {
            return Span<const float>(n.point.data(), n.point.size());
        }
        if (n.q8_data.empty()) {
            return Span<const float>(nullptr, 0);
        }
        const size_t d = n.q8_data.size();
        scratch.resize(d);
        const float range = n.q_max - n.q_min;
        for (size_t i = 0; i < d; ++i) {
            scratch[i] = n.q_min + range * (static_cast<float>(n.q8_data[i]) / 255.0f);
        }
        return Span<const float>(scratch.data(), scratch.size());
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
        // C5a: removed deleted_ids_.erase(nodeId). Re-insert of a tombstoned id
        // is now routed by LSMVecDB::Insert (C5b) to allocate a fresh
        // internal_id via Update semantics; this function never sees a
        // tombstoned nodeId in the new design.
        ++total_inserts_ever_;       // C8: stats
        bool vectorStored = false;  // Track whether we've stored this vector

        auto insert_timer = stats.startTimer();
        int highestLayer = randomLevel();

        if (highestLayer > 0)
        {
            Node newNode;
            newNode.id = nodeId;
            newNode.point = vector;
            // 20260516_upper_layer_sq8: quantize destructively at construction.
            // Frees the float buffer; q8_data carries the dim-byte representation.
            quantizeNodePoint(newNode);
            nodes_[nodeId] = std::move(newNode);
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

        // ----- Greedy descent from top layer to (highestLayer+1) -----
        node_id_t sectionKey = entry_point_;
        for (int l = max_layer_; l > highestLayer; --l)
        {
            currentEntryPoint = greedySearchUpperLayer(vector, currentEntryPoint, l);
            if (l == section_layer_) {
                sectionKey = currentEntryPoint;
            }
        }

        // ----- From min(max_layer_, highestLayer) ... down to 0 -----
        thread_local SearchScratch insert_scratch;
        for (int l = std::min(max_layer_, highestLayer); l >= 0; --l)
        {
            std::vector<SearchResult> neighbors =
                searchLayer(vector, currentEntryPoint, ef_construction_, l, insert_scratch);
            std::vector<node_id_t> selectedNeighbors =
                selectNeighbors(vector, neighbors, m_, l);

            // Refine sectionKey at the adaptive section layer.
            if (l == section_layer_)
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

            // Link neighbors
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
                for (node_id_t neighbor : selectedNeighbors)
                {
                    auto neighborIt = nodes_.find(neighbor);
                    if (neighborIt == nodes_.end()) {
                        LOG(WARN) << "Skipping shrink connections for missing node " << neighbor
                                  << " at layer " << l;
                        continue;
                    }

                    std::vector<node_id_t> eConn = neighborIt->second.neighbors[l];
                    if (eConn.size() > static_cast<size_t>(m_max_))
                    {
                        // 20260516_upper_layer_sq8: selectNeighbors takes
                        // const vector<float>&; if this neighbor is SQ8 we
                        // have to materialize.
                        std::vector<float> neighborPt;
                        const std::vector<float>* basis = nullptr;
                        if (!neighborIt->second.point.empty()) {
                            basis = &neighborIt->second.point;
                        } else if (dequantizeNodePoint(neighborIt->second, neighborPt)) {
                            basis = &neighborPt;
                        }
                        if (basis == nullptr) continue;
                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(*basis, eConn, m_max_, l);
                        neighborIt->second.neighbors[l] = std::move(eNewConn);
                    }
                }
            }
            else // l == 0
            {
                std::vector<std::pair<rocksdb::node_id_t, rocksdb::node_id_t>> edgesToDelete;

                for (node_id_t neighbor : selectedNeighbors)
                {
                    auto eConns = getEdgesCached(neighbor);

                    if (eConns.size() > static_cast<size_t>(m_max_))
                    {
                        std::vector<float> neighborVector;
                        readVectorWithStats(neighbor, neighborVector);

                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(neighborVector, eConns, m_max_, l);

                        for (auto node : eConns)
                        {
                            if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                            {
                                edgesToDelete.emplace_back(
                                    static_cast<rocksdb::node_id_t>(neighbor),
                                    static_cast<rocksdb::node_id_t>(node));
                            }
                        }
                    }
                }

                if (!edgesToDelete.empty()) {
                    db_->DeleteEdgeBatch(edgesToDelete);
                    // Invalidate cache for affected nodes
                    if (edge_cache_) {
                        for (const auto& [from, to] : edgesToDelete) {
                            edge_cache_->erase(from);
                            edge_cache_->erase(to);
                        }
                    }
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

    // C5a pivot: deleteNode is a tombstone insertion only. The HNSW graph is
    // intentionally left untouched (no edge removal, no nodes_ erase, no
    // vector_storage_->deleteVector call); deleted nodes remain valid routing
    // infrastructure per docs/DELETE_DESIGN.md §1.3. Search filters them out
    // at the result-emission step (see knnSearchK below).
    Status LSMVec::deleteNode(node_id_t id)
    {
        tombstoned_internal_ids_.insert(id);
        return Status::OK();
    }

    // updateNode was removed in C5a. Update is now composed at the LSMVecDB
    // layer (C5b): allocate a fresh internal_id, insertNode at it, tombstone
    // the old internal_id. See docs/DELETE_DESIGN.md §5.2.

    Status LSMVec::getNodeVector(node_id_t id, std::vector<float>* out)
    {
        if (!out) {
            return Status::InvalidArgument("output vector must not be null");
        }
        if (tombstoned_internal_ids_.count(id) > 0) {
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

        // Per-thread scratch: capacity warmed across calls within one
        // worker thread, but never shared across threads. See
        // doc/thread-safety-refactor-plan.md (Step 1).
        thread_local SearchScratch scratch;

        auto search_timer = stats.startTimer();
        node_id_t currentEntryPoint = entry_point_;
        for (int l = max_layer_; l >= 1; --l) {
            currentEntryPoint = greedySearchUpperLayer(query, currentEntryPoint, l);
        }

        int ef = std::max(ef_search, k);
        std::vector<SearchResult> neighbors =
            searchLayer(query, currentEntryPoint, ef, 0, scratch);
        if (neighbors.empty()) {
            return neighbors;
        }

        std::vector<SearchResult> filtered;
        filtered.reserve(static_cast<size_t>(k));
        for (const auto& result : neighbors) {
            if (tombstoned_internal_ids_.count(result.id) > 0) {
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

    std::vector<SearchResult> LSMVec::knnSearchKFiltered(
        const std::vector<float>& query,
        int k,
        int ef_search,
        const metadata::Predicate* pred,
        int max_scan_candidates,
        const MetadataStore* meta_store)
    {
        if (entry_point_ == k_invalid_node_id || k <= 0) {
            return {};
        }

        thread_local SearchScratch scratch;

        // Greedy descent through upper layers (filter-blind).
        node_id_t currentEntryPoint = entry_point_;
        for (int lvl = max_layer_; lvl > 0; --lvl) {
            currentEntryPoint = greedySearchUpperLayer(query, currentEntryPoint, lvl);
        }

        // Layer-0 filtered iterative expansion.
        return searchLayer(query, currentEntryPoint, ef_search, /*layer=*/0,
                           pred, k, max_scan_candidates, meta_store, scratch);
    }

    // Links neighbors for upper layers stored in memory
    void LSMVec::linkNeighbors(node_id_t nodeId, const std::vector<node_id_t> &neighborIds, int layer)
    {
        auto nodeIt = nodes_.find(nodeId);
        if (nodeIt == nodes_.end()) {
            LOG(WARN) << "Skipping upper-layer link for missing node " << nodeId
                      << " at layer " << layer;
            return;
        }

        for (node_id_t neighborId : neighborIds)
        {
            auto neighborIt = nodes_.find(neighborId);
            if (neighborIt == nodes_.end()) {
                LOG(WARN) << "Skipping upper-layer link to missing neighbor " << neighborId
                          << " at layer " << layer;
                continue;
            }
            neighborIt->second.neighbors[layer].push_back(nodeId);
            nodeIt->second.neighbors[layer].push_back(neighborId);
        }
    }

    void LSMVec::linkNeighborsAsterDB(node_id_t nodeId, const std::vector<node_id_t> &neighborIds)
    {
        // Create vertex with all forward (out) edges in a single Put.
        // in_neighbors left empty — reverse edges are added individually below.
        std::vector<ROCKSDB_NAMESPACE::node_id_t> out_neighbors(neighborIds.begin(), neighborIds.end());
        std::vector<ROCKSDB_NAMESPACE::node_id_t> in_neighbors;
        db_->AddVertexWithEdges(nodeId, out_neighbors, in_neighbors);

        // Add reverse edges: each neighbor → new node
        for (node_id_t neighborId : neighborIds) {
            db_->AddEdge(neighborId, nodeId);
            if (edge_cache_) edge_cache_->erase(neighborId);
        }
    }

    std::vector<node_id_t> LSMVec::getEdgesCached(node_id_t id) {
        if (edge_cache_) {
            std::vector<node_id_t> cached;
            if (edge_cache_->get(id, &cached)) return cached;
        }

        rocksdb::Edges edges;
        db_->GetAllEdges(id, &edges);
        std::vector<node_id_t> result;
        result.reserve(edges.num_edges_out);
        for (uint32_t i = 0; i < edges.num_edges_out; ++i) {
            result.push_back(edges.nxts_out[i].nxt);
        }
        rocksdb::free_edges(&edges);

        if (edge_cache_) {
            edge_cache_->put(id, result);
        }
        return result;
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
            // 20260516_upper_layer_sq8: scratch for dequantized upper-layer
            // points. Reused across candidates within this select call.
            std::vector<float> sn_simple_scratch;
            std::priority_queue<std::pair<float, node_id_t>> topCandidates;
            for (node_id_t candidateId : candidateIds)
            {
                float dist = 0.0;
                if (layer > 0)
                {
                    const auto nodeIt = nodes_.find(candidateId);
                    bool got = false;
                    if (nodeIt != nodes_.end()) {
                        Span<const float> pt = nodePointView(
                            nodeIt->second, sn_simple_scratch);
                        if (!pt.empty()) {
                            dist = computeDistance(Span<const float>(vector), pt);
                            got = true;
                        }
                    }
                    if (!got) {
                        std::vector<float> candidateVector;
                        readVectorWithStats(candidateId, candidateVector);
                        dist = computeDistance(Span<const float>(vector),
                                               Span<const float>(candidateVector));
                    }
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
        // 20260516_upper_layer_sq8: same cache also memoizes dequantized
        // upper-layer points (where node.point is empty after quantization).
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t nodeId) -> const std::vector<float>& {
            if (layer > 0) {
                auto nodeIt = nodes_.find(nodeId);
                if (nodeIt != nodes_.end()) {
                    if (!nodeIt->second.point.empty()) {
                        return nodeIt->second.point;
                    }
                    auto cit = vecCache.find(nodeId);
                    if (cit != vecCache.end()) return cit->second;
                    std::vector<float> v;
                    if (dequantizeNodePoint(nodeIt->second, v)) {
                        auto res = vecCache.emplace(nodeId, std::move(v));
                        return res.first->second;
                    }
                }
            }

            auto it = vecCache.find(nodeId);
            if (it != vecCache.end()) return it->second;

            std::vector<float> v;
            readVectorWithStats(nodeId, v);

            auto res = vecCache.emplace(nodeId, std::move(v));
            return res.first->second;
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
        // 20260516_upper_layer_sq8: same cache also memoizes dequantized
        // upper-layer points (where node.point is empty after quantization).
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t nodeId) -> const std::vector<float>& {
            if (layer > 0) {
                auto nodeIt = nodes_.find(nodeId);
                if (nodeIt != nodes_.end()) {
                    if (!nodeIt->second.point.empty()) {
                        return nodeIt->second.point;
                    }
                    auto cit = vecCache.find(nodeId);
                    if (cit != vecCache.end()) return cit->second;
                    std::vector<float> v;
                    if (dequantizeNodePoint(nodeIt->second, v)) {
                        auto res = vecCache.emplace(nodeId, std::move(v));
                        return res.first->second;
                    }
                }
            }

            auto it = vecCache.find(nodeId);
            if (it != vecCache.end()) return it->second;

            std::vector<float> v;
            readVectorWithStats(nodeId, v);

            auto res = vecCache.emplace(nodeId, std::move(v));
            return res.first->second;
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

    // --- SearchResult overloads (fast path: skip redundant distance computation) ---

    std::vector<node_id_t> LSMVec::selectNeighbors(
        const std::vector<float>& vector,
        const std::vector<SearchResult>& candidates,
        int maxNeighbors,
        int layer)
    {
        if (!use_heuristic_neighbor_selection_) {
            // Simple: just take the closest maxNeighbors by precomputed distance
            std::vector<SearchResult> sorted(candidates);
            std::sort(sorted.begin(), sorted.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.distance < b.distance;
                      });
            std::vector<node_id_t> result;
            result.reserve(std::min(static_cast<size_t>(maxNeighbors), sorted.size()));
            for (size_t i = 0; i < sorted.size() && static_cast<int>(i) < maxNeighbors; ++i)
                result.push_back(sorted[i].id);
            return result;
        }
        return selectNeighborsHeuristic2(vector, candidates, maxNeighbors, layer);
    }

    std::vector<node_id_t> LSMVec::selectNeighborsHeuristic2(
        const std::vector<float>& vector,
        const std::vector<SearchResult>& candidates,
        int maxNeighbors,
        int layer)
    {
        if (candidates.size() <= static_cast<size_t>(maxNeighbors)) {
            std::vector<node_id_t> result;
            result.reserve(candidates.size());
            for (const auto& c : candidates) result.push_back(c.id);
            return result;
        }

        struct CandidateInfo {
            node_id_t id;
            float distToQuery;
        };

        // 1) Use precomputed distances — no readVector or computeDistance needed
        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidates.size());
        for (const auto& sr : candidates) {
            candInfos.push_back({sr.id, sr.distance});
        }

        // 2) Sort by distance to query
        std::sort(candInfos.begin(), candInfos.end(),
                  [](const CandidateInfo& a, const CandidateInfo& b) {
                      return a.distToQuery < b.distToQuery;
                  });

        const size_t N = candInfos.size();
        const size_t dim = static_cast<size_t>(vector_dim_);

        // 3) Read all candidate vectors into a contiguous buffer
        std::vector<float> candVecs(N * dim);
        if (layer > 0) {
            // 20260516_upper_layer_sq8: route through nodePointView; reuse one
            // scratch buffer across all candidates within this select call.
            std::vector<float> sn2_scratch;
            for (size_t i = 0; i < N; ++i) {
                auto nodeIt = nodes_.find(candInfos[i].id);
                if (nodeIt != nodes_.end()) {
                    Span<const float> pt = nodePointView(nodeIt->second, sn2_scratch);
                    if (pt.size() == dim) {
                        std::memcpy(candVecs.data() + i * dim,
                                    pt.data(), dim * sizeof(float));
                    }
                }
            }
        } else {
            std::vector<node_id_t> ids(N);
            for (size_t i = 0; i < N; ++i) ids[i] = candInfos[i].id;
            vector_storage_->readVectorsBatchFlat(ids, candVecs.data(), dim);
        }

        // 4) Diversity pruning with contiguous buffer (no hash map)
        std::vector<node_id_t> selected;
        selected.reserve(maxNeighbors);
        std::vector<size_t> selectedIndices;
        selectedIndices.reserve(maxNeighbors);

        for (size_t ci = 0; ci < N; ++ci) {
            if (static_cast<int>(selected.size()) >= maxNeighbors)
                break;

            bool good = true;
            const float* candPtr = candVecs.data() + ci * dim;
            for (size_t si : selectedIndices) {
                const float* selPtr = candVecs.data() + si * dim;
                float d = computeDistance(Span<const float>(candPtr, dim),
                                          Span<const float>(selPtr, dim));
                if (d < candInfos[ci].distToQuery) {
                    good = false;
                    break;
                }
            }

            if (good) {
                selected.push_back(candInfos[ci].id);
                selectedIndices.push_back(ci);
            }
        }

        return selected;
    }

    node_id_t LSMVec::greedySearchUpperLayer(const std::vector<float>& query,
                                               node_id_t entryPoint, int layer)
    {
        node_id_t current = entryPoint;
        // 20260516_upper_layer_sq8: dequant scratches — one for the current
        // node, one for each neighbor being scored. Reused across iterations.
        std::vector<float> greedy_scratch_cur;
        std::vector<float> greedy_scratch_neighbor;
        const Node& entryNode = nodes_.at(current);
        Span<const float> entryPt = nodePointView(entryNode, greedy_scratch_cur);
        float bestDist = computeDistance(Span<const float>(query), entryPt);
        bool improved = true;
        while (improved) {
            improved = false;
            auto nodeIt = nodes_.find(current);
            if (nodeIt == nodes_.end()) break;
            auto layerIt = nodeIt->second.neighbors.find(layer);
            if (layerIt == nodeIt->second.neighbors.end()) break;
            for (node_id_t neighborId : layerIt->second) {
                auto nIt = nodes_.find(neighborId);
                if (nIt == nodes_.end()) continue;
                Span<const float> neighborPt = nodePointView(nIt->second,
                                                              greedy_scratch_neighbor);
                if (neighborPt.empty()) continue;
                float d = computeDistance(Span<const float>(query), neighborPt);
                if (d < bestDist) {
                    bestDist = d;
                    current = neighborId;
                    improved = true;
                }
            }
        }
        return current;
    }

    std::vector<SearchResult> LSMVec::searchLayer(const std::vector<float>& queryVector,
                                                  node_id_t entryPointId,
                                                  int efSearch,
                                                  int layer,
                                                  SearchScratch& scratch)
    {
        if (layer < 0) {
            LOG(ERR) << "Invalid layer for search";
            return {};
        }
        if (efSearch <= 0) {
            return {};
        }

        // Versioned visited map (reused across calls inside one scratch,
        // zero allocation after warmup). Wraparound at uint32_t::max
        // forces a one-time clear.
        ++scratch.visited_version;
        if (scratch.visited_version == 0) {
            scratch.visited_map.clear();
            scratch.visited_version = 1;
        }
        auto visitedInsert = [&](node_id_t id) -> bool {
            auto [it, inserted] = scratch.visited_map.emplace(id, scratch.visited_version);
            if (inserted || it->second != scratch.visited_version) {
                it->second = scratch.visited_version;
                return true;  // newly visited
            }
            return false;  // already visited
        };

        // Candidates: max-heap by (-distance) => smallest distance comes first
        using Cand = std::pair<float, node_id_t>;
        std::priority_queue<Cand> candidates;

        // W: max-heap by (distance) => farthest among current best is on top
        std::priority_queue<Cand> nearest;

        // 20260516_upper_layer_sq8: dequant scratch for the upper-layer
        // getDistance path. Reused across calls within this searchLayer.
        std::vector<float> sl_scratch;
        auto getDistance = [&](node_id_t nodeId) -> float {
            if (layer > 0) {
                // Upper layers: vectors are in-memory when available.
                const auto nodeIt = nodes_.find(nodeId);
                if (nodeIt != nodes_.end()) {
                    Span<const float> pt = nodePointView(nodeIt->second, sl_scratch);
                    if (pt.size() == static_cast<size_t>(vector_dim_)) {
                        return computeDistance(Span<const float>(queryVector), pt);
                    }
                }
                // Fallback to disk if the node is not tracked in upper layers.
                std::vector<float> v;
                readVectorWithStats(nodeId, v);
                return computeDistance(Span<const float>(queryVector),
                                       Span<const float>(v));
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
        visitedInsert(entryPointId);
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
                std::vector<float> sl2_scratch;
                for (node_id_t neighborId : neighborIds) {
                    if (visitedInsert(neighborId)) {
                        const auto neighborIt = nodes_.find(neighborId);
                        float d = 0.0f;
                        bool got = false;
                        if (neighborIt != nodes_.end()) {
                            // 20260516_upper_layer_sq8: route through nodePointView
                            // so SQ8-resident neighbors dequant on demand.
                            Span<const float> pt = nodePointView(neighborIt->second, sl2_scratch);
                            if (pt.size() == static_cast<size_t>(vector_dim_)) {
                                d = computeDistance(Span<const float>(queryVector), pt);
                                got = true;
                            }
                        }
                        if (!got) {
                            std::vector<float> neighborVec;
                            readVectorWithStats(neighborId, neighborVec);
                            d = computeDistance(Span<const float>(queryVector),
                                                Span<const float>(neighborVec));
                        }
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
                auto neighborList = getEdgesCached(currentId);

                if (enable_batch_read_) {
                    // --- Batch read path: collect unvisited, read all at once ---
                    std::vector<node_id_t> unvisitedIds;
                    unvisitedIds.reserve(neighborList.size());
                    for (node_id_t neighborId : neighborList) {
                        if (visitedInsert(neighborId)) {
                            unvisitedIds.push_back(neighborId);
                        }
                    }

                    if (!unvisitedIds.empty()) {
                        scratch.batch_read_buf.resize(unvisitedIds.size() * static_cast<size_t>(vector_dim_));
                        vector_storage_->readVectorsBatchFlat(
                            unvisitedIds, scratch.batch_read_buf.data(),
                            static_cast<size_t>(vector_dim_));

                        for (size_t i = 0; i < unvisitedIds.size(); ++i) {
                            float d = computeDistance(
                                Span<const float>(queryVector),
                                Span<const float>(scratch.batch_read_buf.data() + i * static_cast<size_t>(vector_dim_),
                                                  static_cast<size_t>(vector_dim_)));
                            if (static_cast<int>(nearest.size()) < efSearch || d < nearest.top().first) {
                                candidates.emplace(-d, unvisitedIds[i]);
                                nearest.emplace(d, unvisitedIds[i]);
                                if (static_cast<int>(nearest.size()) > efSearch) {
                                    nearest.pop();
                                }
                            }
                        }
                    }
                } else {
                    // --- Original path: read one-by-one ---
                    for (node_id_t neighborId : neighborList) {
                        if (visitedInsert(neighborId)) {
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
                }
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

    std::vector<SearchResult> LSMVec::searchLayer(
        const std::vector<float>& queryVector,
        node_id_t entryPointId,
        int efSearch,
        int layer,
        const metadata::Predicate* pred,
        int k,
        int max_scan_candidates,
        const MetadataStore* meta_store,
        SearchScratch& scratch) {

        (void)efSearch;  // In filter mode, k + max_scan_candidates control termination.

        // No predicate → unfiltered path.
        if (pred == nullptr) {
            return searchLayer(queryVector, entryPointId, efSearch, layer, scratch);
        }
        // Filtered searchLayer is only supported on layer 0.
        assert(layer == 0 && "filtered searchLayer only supported on layer 0");

        if (k <= 0) {
            return {};
        }
        if (max_scan_candidates <= 0) {
            max_scan_candidates = k * 50;
        }

        // Bump visited version (same pattern as the unfiltered overload).
        ++scratch.visited_version;
        if (scratch.visited_version == 0) {
            scratch.visited_map.clear();
            scratch.visited_version = 1;
        }
        auto visitedInsert = [&](node_id_t id) -> bool {
            auto [it, inserted] = scratch.visited_map.emplace(id, scratch.visited_version);
            if (inserted || it->second != scratch.visited_version) {
                it->second = scratch.visited_version;
                return true;  // newly visited
            }
            return false;  // already visited
        };

        // Helper: fetch & parse metadata for node id.
        // Returns an empty object if the row is NotFound (so nothing matches
        // unless the predicate is empty); std::nullopt on any other store
        // error → treated as a non-match for safety.
        auto fetch_meta = [&](node_id_t id) -> std::optional<metadata::Json> {
            if (meta_store == nullptr) return metadata::Json::object();
            metadata::Json j;
            stats.addCount(1, stats.metadata_gets);
            auto st = meta_store->Get(id, &j);
            if (st.IsNotFound()) return metadata::Json::object();
            if (!st.ok()) return std::nullopt;
            return j;
        };
        auto node_matches = [&](node_id_t id) -> bool {
            auto doc = fetch_meta(id);
            if (!doc.has_value()) return false;
            stats.addCount(1, stats.filter_evaluations);
            bool m = pred->matches(*doc);
            if (m) stats.addCount(1, stats.filter_matches);
            return m;
        };

        // candidates: min-heap by distance (stored as max-heap of -distance).
        using Cand = std::pair<float, node_id_t>;
        std::priority_queue<Cand> candidates;   // key = -distance → top is nearest
        std::priority_queue<Cand> results;      // key = distance  → top is farthest
        const size_t kcap = static_cast<size_t>(k);

        // Seed with entry point — routing always, filter only for results.
        std::vector<float> entry_vec;
        readVectorWithStats(entryPointId, entry_vec);
        float d_entry = computeDistance(Span<const float>(queryVector),
                                        Span<const float>(entry_vec));
        visitedInsert(entryPointId);
        candidates.emplace(-d_entry, entryPointId);
        if (node_matches(entryPointId)) {
            results.emplace(d_entry, entryPointId);
        }

        size_t scanned = 0;
        bool cap_hit = false;
        while (!candidates.empty()) {
            float cd = -candidates.top().first;
            node_id_t cid = candidates.top().second;
            candidates.pop();

            // Convergence: no reachable point can improve top-k.
            if (results.size() >= kcap && cd > results.top().first) {
                break;
            }
            // Expansion cap (Tier B safety bound).
            if (scanned >= static_cast<size_t>(max_scan_candidates)) {
                cap_hit = true;
                break;
            }

            // Layer-0 adjacency.
            auto neighborList = getEdgesCached(cid);
            for (node_id_t nb : neighborList) {
                if (!visitedInsert(nb)) continue;

                std::vector<float> nb_vec;
                readVectorWithStats(nb, nb_vec);
                float d = computeDistance(Span<const float>(queryVector),
                                          Span<const float>(nb_vec));
                ++scanned;

                // Routing: filter-agnostic (all neighbors can serve as bridges).
                if (results.size() < kcap || d < results.top().first) {
                    candidates.emplace(-d, nb);
                }

                // Filtering: only matching nodes enter results.
                if (node_matches(nb)) {
                    if (results.size() < kcap || d < results.top().first) {
                        results.emplace(d, nb);
                        if (results.size() > kcap) results.pop();
                    }
                }
            }
        }

        // Drain max-heap → sorted ascending by distance.
        std::vector<SearchResult> out;
        out.reserve(results.size());
        while (!results.empty()) {
            out.push_back({results.top().second, results.top().first});
            results.pop();
        }
        std::reverse(out.begin(), out.end());

        // Record filter-path stats (no-op when stats are disabled).
        stats.addCount(scanned, stats.filter_scanned);
        if (cap_hit) stats.addCount(1, stats.filter_cap_hits);

        return out;
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
        thread_local SearchScratch scratch;
        auto search_timer = stats.startTimer();
        // W ← ∅ set for the current nearest elements
        std::vector<SearchResult> nearestNeighbors; // W: dynamic list of found nearest neighbors

        // ep ← get enter point for hnsw
        node_id_t currentEntryPoint = entry_point_;
        int currentLayer = max_layer_;

        for (int l = max_layer_; l >= 1; --l)
        {
            std::vector<SearchResult> nearestNeighbors =
                searchLayer(queryVector, currentEntryPoint, 30, l, scratch);
            currentEntryPoint = nearestNeighbors[0].id;
        }
        nearestNeighbors = searchLayer(queryVector, currentEntryPoint, 30, 0, scratch);

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
        vector_storage_->flushWrites();
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

    // ----------------------------------------------------------------------
    // Delete / update primitives — see docs/DELETE_DESIGN.md §4 and §5.6.
    // Wired into LSMVecDB in chunk C5b; defined here so resolver tests in
    // C3 can exercise them in isolation.
    // ----------------------------------------------------------------------

    internal_id_t LSMVec::resolve_internal(real_id_t r) const {
        if (!update_bloom_.might_contain(r)) {
            return static_cast<internal_id_t>(r);                  // 99% case
        }
        auto it = updated_real_to_internal_.find(r);
        return (it == updated_real_to_internal_.end())
             ? static_cast<internal_id_t>(r)                       // bloom FP fallback
             : it->second;
    }

    real_id_t LSMVec::resolve_real(internal_id_t i) const {
        if (is_direct_id(i)) return static_cast<real_id_t>(i);     // bit-test fast path
        auto it = updated_internal_to_real_.find(i);
        return (it == updated_internal_to_real_.end())
             ? static_cast<real_id_t>(i)                           // defensive; should not happen
             : it->second;
    }

    bool LSMVec::is_alive(real_id_t r) const {
        internal_id_t i = resolve_internal(r);
        if (tombstoned_internal_ids_.count(i) > 0) return false;
        return vector_storage_ && vector_storage_->exists(i);
    }

    void LSMVec::record_update_mapping(real_id_t r, internal_id_t new_i) {
        // A4: if r already has an update mapping, drop the stale reverse entry.
        auto prev = updated_real_to_internal_.find(r);
        if (prev != updated_real_to_internal_.end()) {
            updated_internal_to_real_.erase(prev->second);
        }
        updated_real_to_internal_[r]     = new_i;
        updated_internal_to_real_[new_i] = r;
        update_bloom_.add(r);
        if (update_bloom_.fill_ratio() >= 0.7) {
            rebuild_bloom_to(update_bloom_.capacity() * 2);
        }
    }

    void LSMVec::rebuild_bloom_to(std::size_t new_capacity) {
        update_bloom_.reset(new_capacity, 0.01);
        for (const auto& kv : updated_real_to_internal_) {
            update_bloom_.add(kv.first);
        }
        ++bloom_rebuild_count_;  // C8: stats
    }

    void LSMVec::forget_update_mapping(real_id_t r) {
        auto it = updated_real_to_internal_.find(r);
        if (it != updated_real_to_internal_.end()) {
            updated_internal_to_real_.erase(it->second);
            updated_real_to_internal_.erase(it);
        }
        // Bloom filter bit stays set; FP is harmless (falls through to the
        // now-empty hashmap and returns identity).
    }

} // namespace ROCKSDB_NAMESPACE
