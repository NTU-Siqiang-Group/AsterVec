#include "metadata_store.h"

#include <cstring>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/write_batch.h"

namespace astervec {

using ROCKSDB_NAMESPACE::PinnableSlice;
using ROCKSDB_NAMESPACE::ReadOptions;
using ROCKSDB_NAMESPACE::Slice;
using ROCKSDB_NAMESPACE::WriteOptions;

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
