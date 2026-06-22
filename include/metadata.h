#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "rocksdb/status.h"

namespace astervec::metadata {

using Json      = nlohmann::json;
using FieldPath = std::vector<std::string>;
using Status    = ROCKSDB_NAMESPACE::Status;

class Predicate {
public:
    enum class Kind {
        Eq, Ne, Gt, Gte, Lt, Lte,
        In, Nin,
        Exists,
        ContainsAny, ContainsAll,
        And, Or,
    };

    Kind                    kind = Kind::And;
    FieldPath               path;
    Json                    value;
    std::vector<Json>       values;
    bool                    exists_expected = true;
    std::vector<Predicate>  children;

    bool matches(const Json& doc) const;
};

// Parse a Mongo-style filter JSON string into a Predicate.
// Empty string or "{}" yields an empty PredAnd that matches everything.
Status ParsePredicate(std::string_view json_str, Predicate* out);

}  // namespace astervec::metadata
