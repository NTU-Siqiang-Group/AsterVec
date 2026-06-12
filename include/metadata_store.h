#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "disk_vector.h"  // provides lsm_vec::node_id_t
#include "json.hpp"
#include "rocksdb/db.h"
#include "rocksdb/status.h"

namespace lsm_vec {

class MetadataStore {
public:
    using Json   = nlohmann::json;
    using Status = ROCKSDB_NAMESPACE::Status;
    using DB     = ROCKSDB_NAMESPACE::DB;
    using ColumnFamilyHandle = ROCKSDB_NAMESPACE::ColumnFamilyHandle;

    // Caller owns db. MetadataStore does not own the CF handle (the LSMVecDB
    // that created the CF does), it only borrows both.
    MetadataStore(DB* db, ColumnFamilyHandle* cf);

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

    // Observability counters (incremented on each call).
    size_t point_gets() const { return point_gets_; }
    size_t parses()     const { return parses_; }

private:
    DB*                 db_;
    ColumnFamilyHandle* cf_;
    mutable size_t      point_gets_ = 0;
    mutable size_t      parses_     = 0;

    static std::string EncodeKey(node_id_t id);  // big-endian uint64
};

}  // namespace lsm_vec
