#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/options.h"

namespace astervec::test {

// Opens a RocksDB at a unique temp directory with a "metadata" CF.
// Destructor closes the DB and removes the directory.
struct TempDB {
    std::string path;
    ROCKSDB_NAMESPACE::DB* db = nullptr;
    ROCKSDB_NAMESPACE::ColumnFamilyHandle* cf_default = nullptr;
    ROCKSDB_NAMESPACE::ColumnFamilyHandle* cf_metadata = nullptr;

    TempDB() {
        char tmpl[] = "/tmp/astervec_mdstore_XXXXXX";
        char* dir = mkdtemp(tmpl);
        path = dir ? dir : "";

        ROCKSDB_NAMESPACE::Options options;
        options.create_if_missing = true;
        options.create_missing_column_families = true;

        std::vector<ROCKSDB_NAMESPACE::ColumnFamilyDescriptor> cfds;
        cfds.emplace_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName,
                          ROCKSDB_NAMESPACE::ColumnFamilyOptions());
        cfds.emplace_back("metadata", ROCKSDB_NAMESPACE::ColumnFamilyOptions());

        std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle*> handles;
        auto st = ROCKSDB_NAMESPACE::DB::Open(options, path, cfds, &handles, &db);
        if (!st.ok()) { std::fprintf(stderr, "TempDB open failed: %s\n", st.ToString().c_str()); std::abort(); }
        cf_default  = handles[0];
        cf_metadata = handles[1];
    }

    ~TempDB() {
        if (db) {
            db->DestroyColumnFamilyHandle(cf_default);
            db->DestroyColumnFamilyHandle(cf_metadata);
            delete db;
        }
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    }
};

}  // namespace astervec::test
