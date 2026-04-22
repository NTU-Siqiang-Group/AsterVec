#include "metadata.h"

#include <cctype>
#include <string>

namespace lsm_vec::metadata {

namespace {

// Hard caps on untrusted filter input. Rejected before json::parse or
// before deep AST construction/traversal, to prevent DoS via
// pathologically large or nested payloads. See
// docs/METADATA_FILTERING_FOLLOWUP.md (I1/M5).
constexpr size_t kMaxFilterJsonBytes = 8 * 1024;
constexpr int    kMaxPredicateDepth  = 32;

// Resolve a dot-path against a JSON document.
// Returns nullptr if any segment is missing or its parent is not an object.
const Json* Resolve(const Json& doc, const FieldPath& path) {
    const Json* cur = &doc;
    for (const auto& seg : path) {
        if (!cur->is_object()) return nullptr;
        auto it = cur->find(seg);
        if (it == cur->end()) return nullptr;
        cur = &*it;
    }
    return cur;
}

// True if a and b are both numbers OR both strings (same JSON ordering category).
bool SameOrderingKind(const Json& a, const Json& b) {
    return (a.is_number() && b.is_number()) ||
           (a.is_string() && b.is_string());
}

template <typename Cmp>
bool CompareOrdered(const Json* v, const Json& rhs, Cmp cmp) {
    if (v == nullptr) return false;
    if (!SameOrderingKind(*v, rhs)) return false;
    return cmp(*v, rhs);
}

// Split "a.b.c" into ["a", "b", "c"]. Empty string -> empty path.
FieldPath SplitPath(std::string_view key) {
    FieldPath out;
    size_t start = 0;
    for (size_t i = 0; i <= key.size(); ++i) {
        if (i == key.size() || key[i] == '.') {
            out.emplace_back(std::string(key.substr(start, i - start)));
            start = i + 1;
        }
    }
    return out;
}

// Forward declaration used by ParseLogicalArray.
Status ParseObject(const Json& obj, Predicate* out, int depth);

Status DepthLimitExceeded() {
    return Status::InvalidArgument(
        "filter predicate nesting exceeds " +
        std::to_string(kMaxPredicateDepth) + " levels");
}

// Convert an operator-object like {"$gt":5,"$lt":10} on `path` to a Predicate.
// Multiple operator keys -> Predicate::Kind::And with one child per operator.
Status ParseOperatorObject(const FieldPath& path, const Json& obj, Predicate* out) {
    if (!obj.is_object()) {
        return Status::InvalidArgument("expected operator object");
    }
    std::vector<Predicate> leaves;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string& op = it.key();
        const Json& arg = it.value();
        Predicate leaf;
        leaf.path = path;

        if      (op == "$eq")  { leaf.kind = Predicate::Kind::Eq;  leaf.value = arg; }
        else if (op == "$ne")  { leaf.kind = Predicate::Kind::Ne;  leaf.value = arg; }
        else if (op == "$gt")  { leaf.kind = Predicate::Kind::Gt;  leaf.value = arg; }
        else if (op == "$gte") { leaf.kind = Predicate::Kind::Gte; leaf.value = arg; }
        else if (op == "$lt")  { leaf.kind = Predicate::Kind::Lt;  leaf.value = arg; }
        else if (op == "$lte") { leaf.kind = Predicate::Kind::Lte; leaf.value = arg; }
        else if (op == "$in" || op == "$nin" ||
                 op == "$contains_any" || op == "$contains_all") {
            if (!arg.is_array()) {
                return Status::InvalidArgument("operator " + op + " requires an array");
            }
            if      (op == "$in")           leaf.kind = Predicate::Kind::In;
            else if (op == "$nin")          leaf.kind = Predicate::Kind::Nin;
            else if (op == "$contains_any") leaf.kind = Predicate::Kind::ContainsAny;
            else                            leaf.kind = Predicate::Kind::ContainsAll;
            for (const auto& el : arg) leaf.values.push_back(el);
        }
        else if (op == "$exists") {
            if (!arg.is_boolean()) {
                return Status::InvalidArgument("$exists requires a boolean");
            }
            leaf.kind = Predicate::Kind::Exists;
            leaf.exists_expected = arg.get<bool>();
        }
        else {
            return Status::InvalidArgument("unknown operator: " + op);
        }
        leaves.push_back(std::move(leaf));
    }

    if (leaves.size() == 1) {
        *out = std::move(leaves.front());
    } else {
        out->kind = Predicate::Kind::And;
        out->children = std::move(leaves);
    }
    return Status::OK();
}

Status ParseLogicalArray(const Json& arr, Predicate::Kind kind, Predicate* out, int depth) {
    if (depth > kMaxPredicateDepth) return DepthLimitExceeded();
    if (!arr.is_array()) {
        return Status::InvalidArgument("$and/$or requires an array");
    }
    out->kind = kind;
    for (const auto& el : arr) {
        if (!el.is_object()) {
            return Status::InvalidArgument("$and/$or element must be an object");
        }
        Predicate child;
        auto st = ParseObject(el, &child, depth + 1);
        if (!st.ok()) return st;
        out->children.push_back(std::move(child));
    }
    return Status::OK();
}

Status ParseObject(const Json& obj, Predicate* out, int depth) {
    if (depth > kMaxPredicateDepth) return DepthLimitExceeded();
    if (!obj.is_object()) {
        return Status::InvalidArgument("filter must be an object");
    }

    std::vector<Predicate> conjuncts;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string& key = it.key();
        const Json& val = it.value();
        Predicate child;

        if (key == "$and" || key == "$or") {
            auto st = ParseLogicalArray(
                val,
                key == "$and" ? Predicate::Kind::And : Predicate::Kind::Or,
                &child,
                depth + 1);
            if (!st.ok()) return st;
        } else {
            FieldPath path = SplitPath(key);

            // Discriminate operator-object from value-object by inspecting keys:
            // an operator-object has ALL keys starting with '$'. A value-object
            // (non-operator) is treated as implicit $eq with deep-equality.
            bool is_operator_object = false;
            bool has_op = false;
            bool has_non_op = false;
            if (val.is_object() && !val.empty()) {
                is_operator_object = true;
                for (auto vit = val.begin(); vit != val.end(); ++vit) {
                    if (!vit.key().empty() && vit.key()[0] == '$') {
                        has_op = true;
                    } else {
                        has_non_op = true;
                        is_operator_object = false;
                    }
                }
            }

            // Mixed $-prefixed and non-$-prefixed keys almost certainly indicate
            // a user typo rather than a legitimate implicit-eq value. Reject
            // loudly with a clear message (addresses reviewer M2).
            if (has_op && has_non_op) {
                return Status::InvalidArgument(
                    std::string("operator object for field '") + key +
                    "' must contain only $-prefixed keys");
            }

            if (is_operator_object) {
                auto st = ParseOperatorObject(path, val, &child);
                if (!st.ok()) return st;
            } else {
                // implicit eq (scalar, array, empty object, or non-operator object)
                child.kind  = Predicate::Kind::Eq;
                child.path  = std::move(path);
                child.value = val;
            }
        }
        conjuncts.push_back(std::move(child));
    }

    if (conjuncts.size() == 1) {
        *out = std::move(conjuncts.front());
    } else {
        out->kind = Predicate::Kind::And;
        out->children = std::move(conjuncts);
    }
    return Status::OK();
}

// Depth-aware matcher. The public Predicate::matches is a thin wrapper
// that enters with depth = 0; $and/$or arms recurse with depth + 1.
// Above kMaxPredicateDepth the predicate is treated as non-match — a
// safety valve against hand-built over-deep ASTs that bypassed the
// parser's depth cap (M5).
bool MatchesWithDepth(const Predicate& p, const Json& doc, int depth) {
    if (depth > kMaxPredicateDepth) return false;

    const Json* v = Resolve(doc, p.path);
    switch (p.kind) {
        case Predicate::Kind::Eq:
            return v != nullptr && *v == p.value;
        case Predicate::Kind::Ne:
            return v == nullptr || *v != p.value;  // missing → true
        case Predicate::Kind::Gt:
            return CompareOrdered(v, p.value, [](auto& a, auto& b) { return a >  b; });
        case Predicate::Kind::Gte:
            return CompareOrdered(v, p.value, [](auto& a, auto& b) { return a >= b; });
        case Predicate::Kind::Lt:
            return CompareOrdered(v, p.value, [](auto& a, auto& b) { return a <  b; });
        case Predicate::Kind::Lte:
            return CompareOrdered(v, p.value, [](auto& a, auto& b) { return a <= b; });
        case Predicate::Kind::In: {
            if (v == nullptr) return false;
            for (const auto& x : p.values) if (*v == x) return true;
            return false;
        }
        case Predicate::Kind::Nin: {
            if (v == nullptr) return true;
            for (const auto& x : p.values) if (*v == x) return false;
            return true;
        }
        case Predicate::Kind::Exists:
            return (v != nullptr) == p.exists_expected;
        case Predicate::Kind::ContainsAny: {
            if (v == nullptr || !v->is_array()) return false;
            for (const auto& target : p.values)
                for (const auto& el : *v)
                    if (el == target) return true;
            return false;
        }
        case Predicate::Kind::ContainsAll: {
            if (v == nullptr || !v->is_array()) return false;
            for (const auto& target : p.values) {
                bool found = false;
                for (const auto& el : *v)
                    if (el == target) { found = true; break; }
                if (!found) return false;
            }
            return true;
        }
        case Predicate::Kind::And:
            for (const auto& c : p.children)
                if (!MatchesWithDepth(c, doc, depth + 1)) return false;
            return true;
        case Predicate::Kind::Or:
            for (const auto& c : p.children)
                if (MatchesWithDepth(c, doc, depth + 1)) return true;
            return false;
    }
    return false;  // unreachable; silences compiler non-exhaustive-switch warning
}

}  // namespace

bool Predicate::matches(const Json& doc) const {
    return MatchesWithDepth(*this, doc, 0);
}

Status ParsePredicate(std::string_view json_str, Predicate* out) {
    *out = Predicate{};  // reset to default (empty AND = match all)

    // Trim leading/trailing whitespace
    size_t start = 0, end = json_str.size();
    while (start < end && std::isspace(static_cast<unsigned char>(json_str[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(json_str[end-1]))) --end;
    auto trimmed = json_str.substr(start, end - start);

    if (trimmed.empty() || trimmed == "{}") {
        return Status::OK();  // default empty PredAnd matches all
    }

    if (trimmed.size() > kMaxFilterJsonBytes) {
        return Status::InvalidArgument(
            "filter JSON exceeds " + std::to_string(kMaxFilterJsonBytes) + " bytes");
    }

    Json parsed;
    try {
        parsed = Json::parse(trimmed);
    } catch (const std::exception& e) {
        return Status::InvalidArgument(std::string("filter JSON parse error: ") + e.what());
    }

    return ParseObject(parsed, out, 0);
}

}  // namespace lsm_vec::metadata
