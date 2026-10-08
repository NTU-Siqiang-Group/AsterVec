#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "disk_vector.h"  // provides astervec::node_id_t
#include "json.hpp"
#include "rocksdb/db.h"
#include "rocksdb/status.h"

namespace astervec {

class MetadataStore {
public:
    using Json   = nlohmann::json;
    using Status = ROCKSDB_NAMESPACE::Status;
    using DB     = ROCKSDB_NAMESPACE::DB;
    using ColumnFamilyHandle = ROCKSDB_NAMESPACE::ColumnFamilyHandle;

    // Caller owns db. MetadataStore does not own the CF handle (the AsterVecDB
    // that created the CF does), it only borrows both. (Kept for tests; the
    // engine uses Open() below, which owns everything.)
    MetadataStore(DB* db, ColumnFamilyHandle* cf);

    // Opens (creating if missing) the sibling RocksDB at <dir> with the
    // metadata column family and returns a store that OWNS the DB and the CF
    // handle — the storage-seam form: callers hold no RocksDB types.
    //
    // with_id_map: also open the `ids` column family that maps caller point
    // ids to dense internal ids (stores created with IdMode::kKeyed).
    static Status Open(const std::string& dir,
                       std::unique_ptr<MetadataStore>* out,
                       bool with_id_map = false);

    // Destroys the owned CF handle and DB (no-op for the borrowing ctor).
    ~MetadataStore();

    // Write the raw JSON bytes verbatim under the given id. Overwrites any prior value.
    Status Put(node_id_t id, std::string_view json_bytes);
    Status Put(node_id_t id, const Json& doc);

    // Disambiguates raw-string-literal calls like Put(id, R"({...})") which would
    // otherwise match both std::string_view and nlohmann::json implicit conversions.
    Status Put(node_id_t id, const char* json_bytes) {
        return Put(id, std::string_view(json_bytes));
    }

    // Write many (id, raw-json) rows atomically in a single WriteBatch
    // (all-or-nothing). Used by the bulk-build payload path.
    Status PutBatch(const std::vector<std::pair<node_id_t, std::string>>& items);

    // Read the raw JSON bytes. Returns NotFound if no metadata exists for id.
    Status Get(node_id_t id, std::string* out_bytes) const;
    Status Get(node_id_t id, Json* out_doc) const;

    // Delete the metadata row for id. OK if the row does not exist (idempotent).
    Status Delete(node_id_t id);

    // Apply a JSON merge-patch (RFC 7396). Creates a new row if id had no prior metadata.
    Status Update(node_id_t id, const Json& partial);

    // Remove specified keys from the existing metadata. Missing keys are ignored.
    // Returns NotFound if no metadata exists for id.
    Status DeleteKeys(node_id_t id, const std::vector<std::string>& keys);

    // fsync the payload DB's WAL so every write accepted so far is durable.
    Status SyncWAL();

    // ---- Point id map (keyed stores) ----
    // Keys are PointId::Encode() bytes. Both directions are kept so search
    // results can be translated back. Entries are never removed: a key keeps
    // its dense id for the life of the store.
    bool has_id_map() const { return ids_cf_ != nullptr; }
    // NotFound if the key has no dense id yet.
    Status LookupKey(std::string_view key, node_id_t* dense) const;
    // NotFound if no key maps to `dense`.
    Status ReverseLookup(node_id_t dense, std::string* key) const;
    // Records key <-> dense in one atomic batch (idempotent).
    Status PutKey(std::string_view key, node_id_t dense);
    Status PutKeys(const std::vector<std::pair<std::string, node_id_t>>& items);
    // The largest dense id with a key; *any = false when the map is empty.
    Status MaxMappedId(node_id_t* max_id, bool* any) const;

    // Observability counters (incremented on each call).
    size_t point_gets() const { return point_gets_; }
    size_t parses()     const { return parses_; }

private:
    DB*                 db_;
    ColumnFamilyHandle* cf_;
    ColumnFamilyHandle* ids_cf_ = nullptr;  // only with an id map
    bool                owns_ = false;  // true iff created via Open()
    mutable size_t      point_gets_ = 0;
    mutable size_t      parses_     = 0;

    static std::string EncodeKey(node_id_t id);  // big-endian uint64
};

}  // namespace astervec
