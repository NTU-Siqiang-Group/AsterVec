#include "metadata_store.h"

#include <cstring>
#include <memory>

#include "rocksdb/cache.h"
#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "rocksdb/write_batch.h"

namespace astervec {

using ROCKSDB_NAMESPACE::PinnableSlice;
using ROCKSDB_NAMESPACE::ReadOptions;
using ROCKSDB_NAMESPACE::Slice;
using ROCKSDB_NAMESPACE::WriteOptions;

namespace {
constexpr char kForwardPrefix = 'k';   // 'k' + key       -> dense id (BE 8 bytes)
constexpr char kReversePrefix = 'r';   // 'r' + dense(BE)  -> key
}  // namespace

std::string MetadataStore::EncodeKey(node_id_t id) {
    // Big-endian 8-byte encoding: lexicographic order == numeric order.
    std::string out(8, '\0');
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<char>(id & 0xFFu);
        id >>= 8;
    }
    return out;
}

MetadataStore::MetadataStore(DB* db, ColumnFamilyHandle* cf)
    : db_(db), cf_(cf) {}

MetadataStore::~MetadataStore() {
    if (owns_) {
        if (ids_cf_ && db_) db_->DestroyColumnFamilyHandle(ids_cf_);
        if (cf_ && db_) db_->DestroyColumnFamilyHandle(cf_);
        delete db_;
    }
}

MetadataStore::Status MetadataStore::Open(const std::string& dir,
                                          std::unique_ptr<MetadataStore>* out,
                                          bool with_id_map) {
    ROCKSDB_NAMESPACE::Options meta_opts;
    meta_opts.create_if_missing = true;
    meta_opts.create_missing_column_families = true;
    meta_opts.compression = ROCKSDB_NAMESPACE::kZSTD;
    // Cap total memtable memory across all CFs at 16 MB.
    meta_opts.db_write_buffer_size = 16 << 20;

    // One 16 MB LRU cache shared by default + metadata CFs. Without this,
    // each BlockBasedTableFactory would instantiate its own default cache
    // (32 MB each in current RocksDB). The default CF is never written to
    // so its contribution to the cache stays ~0.
    auto shared_block_cache = ROCKSDB_NAMESPACE::NewLRUCache(16 << 20);

    std::vector<ROCKSDB_NAMESPACE::ColumnFamilyDescriptor> cfds;

    ROCKSDB_NAMESPACE::BlockBasedTableOptions default_table_opts;
    default_table_opts.block_cache = shared_block_cache;
    ROCKSDB_NAMESPACE::ColumnFamilyOptions default_cfopts;
    // Never written to, but set explicitly so it doesn't inherit RocksDB's
    // kSnappy default (Snappy is no longer compiled into Aster).
    default_cfopts.compression = ROCKSDB_NAMESPACE::kNoCompression;
    default_cfopts.table_factory.reset(
        ROCKSDB_NAMESPACE::NewBlockBasedTableFactory(default_table_opts));
    cfds.emplace_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName,
                      default_cfopts);

    ROCKSDB_NAMESPACE::ColumnFamilyOptions cfopts;
    cfopts.compression = ROCKSDB_NAMESPACE::kZSTD;
    cfopts.write_buffer_size = 8 << 20;   // 8 MB per memtable
    cfopts.max_write_buffer_number = 2;   // up to 2 memtables: 2 * 8 MB = 16 MB
    ROCKSDB_NAMESPACE::BlockBasedTableOptions table_opts;
    table_opts.block_cache = shared_block_cache;
    table_opts.filter_policy.reset(
        ROCKSDB_NAMESPACE::NewBloomFilterPolicy(10, false));
    cfopts.table_factory.reset(
        ROCKSDB_NAMESPACE::NewBlockBasedTableFactory(table_opts));
    cfds.emplace_back("metadata", cfopts);
    if (with_id_map) {
        // Point lookups only (by key, by dense id) plus one reverse seek on
        // open; same cache and compression as the payloads.
        cfds.emplace_back("ids", cfopts);
    }

    std::vector<ColumnFamilyHandle*> handles;
    DB* raw = nullptr;
    Status st = DB::Open(meta_opts, dir, cfds, &handles, &raw);
    if (!st.ok()) {
        // Some RocksDB versions populate handles even on partial failure.
        for (auto* h : handles) {
            if (h && raw) raw->DestroyColumnFamilyHandle(h);
        }
        delete raw;
        return st;
    }

    // Default CF is unused; destroy its handle now, keep the metadata CF.
    raw->DestroyColumnFamilyHandle(handles[0]);
    auto store = std::unique_ptr<MetadataStore>(
        new MetadataStore(raw, handles[1]));
    if (with_id_map) store->ids_cf_ = handles[2];
    store->owns_ = true;
    *out = std::move(store);
    return Status::OK();
}

MetadataStore::Status MetadataStore::Put(node_id_t id, std::string_view json_bytes) {
    return db_->Put(WriteOptions(), cf_, EncodeKey(id),
                    Slice(json_bytes.data(), json_bytes.size()));
}

MetadataStore::Status MetadataStore::Put(node_id_t id, const Json& doc) {
    auto s = doc.dump();
    return Put(id, std::string_view(s));
}

MetadataStore::Status MetadataStore::PutBatch(
        const std::vector<std::pair<node_id_t, std::string>>& items) {
    ROCKSDB_NAMESPACE::WriteBatch batch;
    for (const auto& kv : items) {
        auto s = batch.Put(cf_, EncodeKey(kv.first),
                           Slice(kv.second.data(), kv.second.size()));
        if (!s.ok()) return s;
    }
    return db_->Write(WriteOptions(), &batch);
}

MetadataStore::Status MetadataStore::Get(node_id_t id, std::string* out_bytes) const {
    ++point_gets_;
    PinnableSlice pin;
    auto st = db_->Get(ReadOptions(), cf_, EncodeKey(id), &pin);
    if (!st.ok()) return st;
    out_bytes->assign(pin.data(), pin.size());
    return Status::OK();
}

MetadataStore::Status MetadataStore::Get(node_id_t id, Json* out_doc) const {
    std::string bytes;
    auto st = Get(id, &bytes);
    if (!st.ok()) return st;
    ++parses_;
    try {
        *out_doc = Json::parse(bytes);
    } catch (const std::exception& e) {
        return Status::Corruption(std::string("metadata parse error: ") + e.what());
    }
    return Status::OK();
}

MetadataStore::Status MetadataStore::SyncWAL() {
    return db_->SyncWAL();
}

MetadataStore::Status MetadataStore::LookupKey(std::string_view key,
                                               node_id_t* dense) const {
    if (!ids_cf_) return Status::NotSupported("store has no id map");
    std::string k;
    k.reserve(1 + key.size());
    k.push_back(kForwardPrefix);
    k.append(key.data(), key.size());
    PinnableSlice v;
    Status s = db_->Get(ReadOptions(), ids_cf_, k, &v);
    if (!s.ok()) return s;
    if (v.size() != 8) return Status::Corruption("bad id map entry");
    node_id_t id = 0;
    for (size_t i = 0; i < 8; ++i) {
        id = (id << 8) | static_cast<unsigned char>(v.data()[i]);
    }
    *dense = id;
    return Status::OK();
}

MetadataStore::Status MetadataStore::ReverseLookup(node_id_t dense,
                                                   std::string* key) const {
    if (!ids_cf_) return Status::NotSupported("store has no id map");
    std::string k(1, kReversePrefix);
    k += EncodeKey(dense);
    return db_->Get(ReadOptions(), ids_cf_, k, key);
}

MetadataStore::Status MetadataStore::PutKey(std::string_view key, node_id_t dense) {
    return PutKeys({{std::string(key), dense}});
}

MetadataStore::Status MetadataStore::PutKeys(
        const std::vector<std::pair<std::string, node_id_t>>& items) {
    if (!ids_cf_) return Status::NotSupported("store has no id map");
    ROCKSDB_NAMESPACE::WriteBatch batch;
    for (const auto& kv : items) {
        const std::string dense = EncodeKey(kv.second);
        std::string fwd(1, kForwardPrefix);
        fwd += kv.first;
        std::string rev(1, kReversePrefix);
        rev += dense;
        auto s = batch.Put(ids_cf_, fwd, dense);
        if (!s.ok()) return s;
        s = batch.Put(ids_cf_, rev, kv.first);
        if (!s.ok()) return s;
    }
    return db_->Write(WriteOptions(), &batch);
}

MetadataStore::Status MetadataStore::MaxMappedId(node_id_t* max_id, bool* any) const {
    if (!ids_cf_) return Status::NotSupported("store has no id map");
    *any = false;
    *max_id = 0;
    std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> it(
        db_->NewIterator(ReadOptions(), ids_cf_));
    // The reverse range ('r' + 8-byte big-endian id) sorts by id: its last
    // entry holds the largest id.
    std::string upper(1, static_cast<char>(kReversePrefix + 1));
    it->SeekForPrev(upper);
    if (it->Valid() && !it->key().empty() && it->key()[0] == kReversePrefix) {
        if (it->key().size() != 9) return Status::Corruption("bad id map entry");
        node_id_t id = 0;
        for (size_t i = 1; i < 9; ++i) {
            id = (id << 8) | static_cast<unsigned char>(it->key()[i]);
        }
        *max_id = id;
        *any = true;
    }
    return it->status();
}

MetadataStore::Status MetadataStore::Delete(node_id_t id) {
    return db_->Delete(WriteOptions(), cf_, EncodeKey(id));
}

MetadataStore::Status MetadataStore::Update(node_id_t id, const Json& partial) {
    Json existing;
    auto st = Get(id, &existing);
    if (st.IsNotFound()) {
        existing = Json::object();
    } else if (!st.ok()) {
        return st;
    }
    existing.merge_patch(partial);
    return Put(id, existing);
}

MetadataStore::Status MetadataStore::DeleteKeys(node_id_t id,
                                                const std::vector<std::string>& keys) {
    Json existing;
    auto st = Get(id, &existing);
    if (!st.ok()) return st;
    if (!existing.is_object()) return Status::InvalidArgument("metadata is not an object");
    for (const auto& k : keys) existing.erase(k);
    return Put(id, existing);
}

}  // namespace astervec
