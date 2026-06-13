// LSM-Vec HTTP server — endpoints and request handlers.
//
// API surface (matches the design docs §5.3):
//   GET    /health                       — gateway / liveness probe
//   GET    /ready                        — DB-open readiness probe
//   GET    /v1/stats                     — observability snapshot
//   POST   /v1/vectors                   — {id, vector, metadata?}
//   GET    /v1/vectors/:id
//   PUT    /v1/vectors/:id               — upsert (vector only)
//   DELETE /v1/vectors/:id
//   GET    /v1/vectors/:id/payload
//   PUT    /v1/vectors/:id/payload       — set
//   PATCH  /v1/vectors/:id/payload       — RFC 7396 merge
//   POST   /v1/search                    — {vector, k?, ef_search?, filter?}
//
// JSON request/response. All errors as
//   { "error": "...", "code": "..." } with appropriate HTTP status.

#include "lsm_vec_http.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace lsm_vec {

using json = nlohmann::json;

namespace {

constexpr char kBootstrapFileName[] = "lsm_vec_http_bootstrap.json";

// ---------- env helpers ----------
std::string env_or(const char* name, std::string deflt) {
    if (const char* v = std::getenv(name)) return std::string(v);
    return deflt;
}
int env_int(const char* name, int deflt) {
    if (const char* v = std::getenv(name)) {
        try { return std::stoi(v); } catch (...) {}
    }
    return deflt;
}
std::size_t env_size(const char* name, std::size_t deflt) {
    if (const char* v = std::getenv(name)) {
        try { return static_cast<std::size_t>(std::stoull(v)); } catch (...) {}
    }
    return deflt;
}
bool env_bool(const char* name, bool deflt) {
    if (const char* v = std::getenv(name)) {
        std::string s(v);
        for (auto& c : s) c = static_cast<char>(std::tolower(c));
        if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
        if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    }
    return deflt;
}

// ---------- response helpers ----------
void sendError(httplib::Response& res, int status, const std::string& code,
               const std::string& msg) {
    json body = {{"error", msg}, {"code", code}};
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void sendJson(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// ---------- parsing helpers ----------
bool parseId(const std::string& s, node_id_t* out, std::string* err) {
    try {
        std::size_t pos = 0;
        unsigned long long u = std::stoull(s, &pos, 10);
        if (pos != s.size()) { *err = "id must be a positive integer"; return false; }
        *out = static_cast<node_id_t>(u);
        return true;
    } catch (...) {
        *err = "id must be a positive integer";
        return false;
    }
}

bool extractVector(const json& v, std::vector<float>* out, std::string* err) {
    if (!v.is_array()) { *err = "vector must be a JSON array of floats"; return false; }
    out->clear();
    out->reserve(v.size());
    for (const auto& x : v) {
        if (!x.is_number()) { *err = "vector elements must be numbers"; return false; }
        out->push_back(x.get<float>());
    }
    return true;
}

// ---------- request logging ----------
struct ReqTimer {
    std::chrono::high_resolution_clock::time_point t0;
    ReqTimer() : t0(std::chrono::high_resolution_clock::now()) {}
    double ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

void logReq(const httplib::Request& req, int status, double latency_ms,
            const std::string& note = "") {
    // JSON one-line log to stdout (captured by docker logs).
    json j = {
        {"ts", static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count())
                  / 1000.0},
        {"method", req.method},
        {"path", req.path},
        {"status", status},
        {"latency_ms", latency_ms},
        {"remote", req.remote_addr},
    };
    if (!note.empty()) j["note"] = note;
    std::cout << j.dump() << "\n" << std::flush;
}

// ---------- DB bootstrap (first-boot persists dim/metric) ----------
struct BootstrapState {
    int dim;
    std::string metric;
};

bool readBootstrap(const std::string& dir, BootstrapState* out) {
    std::ifstream f(dir + "/" + kBootstrapFileName);
    if (!f.is_open()) return false;
    try {
        json j; f >> j;
        out->dim = j.at("dim").get<int>();
        out->metric = j.at("metric").get<std::string>();
        return true;
    } catch (...) { return false; }
}

bool writeBootstrap(const std::string& dir, const BootstrapState& st,
                    std::string* err) {
    std::ofstream f(dir + "/" + kBootstrapFileName);
    if (!f.is_open()) { *err = "cannot open bootstrap file for write"; return false; }
    json j = {{"dim", st.dim}, {"metric", st.metric}};
    f << j.dump(2) << "\n";
    return true;
}

// ---------- signal handling ----------
// Multi-threaded signal handling: SIGTERM/SIGINT can be delivered to
// any thread in the process. httplib spawns worker threads that don't
// know about our handler, so the default action (terminate) wins.
// Standard Unix fix: block these signals in EVERY thread (including
// main), then dedicate ONE thread to wait via sigwait().
std::atomic<httplib::Server*> g_server_ptr{nullptr};
std::atomic<bool> g_shutting_down{false};

void blockShutdownSignalsInThisThread() {
    sigset_t s;
    sigemptyset(&s);
    sigaddset(&s, SIGTERM);
    sigaddset(&s, SIGINT);
    ::pthread_sigmask(SIG_BLOCK, &s, nullptr);
}

void startShutdownWatcher() {
    // The watcher thread inherits the blocked mask from main (set in
    // RunHttpServer before any other thread spawn) and is the only
    // thread allowed to consume SIGTERM/SIGINT via sigwait().
    std::thread([] {
        sigset_t s;
        sigemptyset(&s);
        sigaddset(&s, SIGTERM);
        sigaddset(&s, SIGINT);
        int signo = 0;
        ::sigwait(&s, &signo);
        std::cerr << "{\"event\":\"signal_received\",\"signo\":" << signo
                  << "}\n" << std::flush;
        g_shutting_down.store(true, std::memory_order_release);
        auto* sv = g_server_ptr.load();
        if (sv) sv->stop();
    }).detach();
}

}  // namespace

// ============================================================
// Env-var configuration parsing
// ============================================================

bool LoadHttpConfigFromEnv(HttpServerConfig* cfg, std::string* err) {
    cfg->port = env_int("LSMVEC_PORT", 8000);
    // Default: 1 thread per container. Per-user-container model
    // handles isolation; no contention within a single container.
    // Override with LSMVEC_HTTP_THREADS=N on dedicated hardware.
    cfg->http_threads = env_int("LSMVEC_HTTP_THREADS", 1);
    cfg->data_dir = env_or("LSMVEC_DATA_DIR", "/data");
    cfg->dim = env_int("LSMVEC_DIM", 0);

    std::string metric = env_or("LSMVEC_METRIC", "l2");
    for (auto& c : metric) c = static_cast<char>(std::tolower(c));
    if      (metric == "l2")     cfg->metric = DistanceMetric::kL2;
    else if (metric == "cosine") cfg->metric = DistanceMetric::kCosine;
    else {
        *err = "LSMVEC_METRIC must be 'l2' or 'cosine' (got '" + metric + "')";
        return false;
    }

    cfg->m                  = env_int("LSMVEC_M", 8);
    cfg->m_max              = env_int("LSMVEC_MMAX", 24);
    cfg->ef_construction    = env_int("LSMVEC_EFC", 32);
    cfg->ef_search_default  = env_int("LSMVEC_EFS_DEFAULT", 128);
    cfg->edge_cache_size    = env_size("LSMVEC_EDGE_CACHE_SIZE", 100000);
    cfg->vec_file_capacity  = env_size("LSMVEC_VEC_FILE_CAPACITY", 10000000);
    cfg->paged_max_cached_pages = env_size("LSMVEC_PAGED_MAX_CACHED_PAGES", 8192);
    cfg->enable_stats       = env_bool("LSMVEC_ENABLE_STATS", false);
    cfg->log_level          = env_or("LSMVEC_LOG_LEVEL", "info");

    // Bootstrap: if data dir already has a bootstrap.json, the dim/metric
    // there is authoritative. Otherwise this is first boot and env dim/
    // metric must be present.
    BootstrapState saved;
    if (readBootstrap(cfg->data_dir, &saved)) {
        // Existing DB: validate that env matches if env was set.
        if (cfg->dim > 0 && cfg->dim != saved.dim) {
            *err = "LSMVEC_DIM (" + std::to_string(cfg->dim) +
                   ") does not match existing DB dim (" +
                   std::to_string(saved.dim) + ")";
            return false;
        }
        cfg->dim = saved.dim;
        std::string saved_metric = saved.metric;
        for (auto& c : saved_metric) c = static_cast<char>(std::tolower(c));
        if (saved_metric != metric) {
            // Allow env to default to l2 silently if the DB says l2.
            if (!(metric == "l2" && std::getenv("LSMVEC_METRIC") == nullptr)) {
                *err = "LSMVEC_METRIC ('" + metric +
                       "') does not match existing DB metric ('" +
                       saved.metric + "')";
                return false;
            }
        }
        if      (saved_metric == "cosine") cfg->metric = DistanceMetric::kCosine;
        else                               cfg->metric = DistanceMetric::kL2;
    } else {
        // First boot. If LSMVEC_DIM was provided (operator pre-init / back-compat),
        // persist it now and the server opens the DB at boot. Otherwise leave the
        // index UNINITIALIZED (cfg->dim stays 0, no bootstrap written): the server
        // boots without a DB and the user calls PUT /v1/index to set the dimension.
        if (cfg->dim > 0) {
            std::string err_w;
            if (!writeBootstrap(cfg->data_dir, {cfg->dim, metric}, &err_w)) {
                *err = err_w;
                return false;
            }
        }
    }
    return true;
}

// ============================================================
// Endpoint handlers
// ============================================================

namespace {

const char* metricToString(DistanceMetric m) {
    return (m == DistanceMetric::kCosine) ? "cosine" : "l2";
}

// Build engine options from the server config. Used both at boot
// (RunHttpServer) and at runtime when PUT /v1/index opens the DB.
LSMVecDBOptions buildOptions(const HttpServerConfig& cfg) {
    LSMVecDBOptions opts;
    opts.dim                  = cfg.dim;
    opts.metric               = cfg.metric;
    opts.m                    = cfg.m;
    opts.m_max                = cfg.m_max;
    opts.ef_construction      = static_cast<float>(cfg.ef_construction);
    opts.ef_search            = cfg.ef_search_default;
    opts.vec_file_capacity    = cfg.vec_file_capacity;
    opts.paged_max_cached_pages = cfg.paged_max_cached_pages;
    opts.vector_storage_type  = 1;
    opts.edge_cache_size      = cfg.edge_cache_size;
    opts.enable_stats         = cfg.enable_stats;
    opts.enable_batch_read    = true;
    opts.reinit               = false;
    opts.vector_file_path     = cfg.data_dir + "/vector.log";
    return opts;
}

// Shared, mutable server state. The DB starts null when the index is
// uninitialized (no dimension yet); PUT /v1/index opens it and DELETE
// /v1/index closes + wipes it. `mu` serializes those structural changes
// and guards `cfg` (dim/metric) mutation. Data handlers take a shared_ptr
// copy under `mu` so the DB stays alive for the request even if a
// concurrent DELETE swaps it out.
struct ServerState {
    std::mutex                  mu;
    std::shared_ptr<LSMVecDB>   db;   // null == uninitialized
    HttpServerConfig            cfg;  // dim/metric set by PUT /v1/index
};

class Handlers {
public:
    explicit Handlers(ServerState* st) : st_(st), cfg_(st->cfg) {}

    void registerRoutes(httplib::Server& s) {
        // Health / readiness.
        s.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            sendJson(res, 200, {{"status", "ok"}});
            logReq(req, 200, t.ms());
        });
        s.Get("/ready", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            bool ok = (dbShared() != nullptr);
            int status = ok ? 200 : 503;
            sendJson(res, status, {{"status", ok ? "ok" : "no db"}});
            logReq(req, status, t.ms());
        });

        // Index lifecycle: set / inspect / delete the dimension.
        s.Put("/v1/index", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleCreateIndex(req, res);
            logReq(req, status, t.ms());
        });
        s.Delete("/v1/index", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleDeleteIndex(req, res);
            logReq(req, status, t.ms());
        });
        s.Get("/v1/index", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleGetIndex(req, res);
            logReq(req, status, t.ms());
        });

        // Stats.
        s.Get("/v1/stats", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            auto sp = requireInit(res);
            if (!sp) { logReq(req, 409, t.ms()); return; }
            auto ds = sp->GetDeleteStats();
            json body = {
                {"tombstones", ds.tombstones},
                {"updated_real_ids", ds.updated_real_ids},
                {"total_inserts_ever", ds.total_inserts_ever},
                {"tombstone_ratio", ds.tombstone_ratio},
                {"bloom_capacity", ds.bloom_capacity},
                {"bloom_fill_ratio", ds.bloom_fill_ratio},
                {"bloom_rebuild_count", ds.bloom_rebuild_count},
            };
            sendJson(res, 200, body);
            logReq(req, 200, t.ms());
        });

        // Insert.
        s.Post("/v1/vectors", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleInsert(req, res);
            logReq(req, status, t.ms());
        });
        s.Post("/v1/vectors/batch",
               [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleInsertBatch(req, res);
            logReq(req, status, t.ms());
        });

        // Get / upsert / delete by id.
        s.Get(R"(/v1/vectors/(\d+))",
              [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleGet(req, res);
            logReq(req, status, t.ms());
        });
        s.Put(R"(/v1/vectors/(\d+))",
              [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handlePut(req, res);
            logReq(req, status, t.ms());
        });
        s.Delete(R"(/v1/vectors/(\d+))",
                 [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleDelete(req, res);
            logReq(req, status, t.ms());
        });

        // Payload.
        s.Get(R"(/v1/vectors/(\d+)/payload)",
              [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleGetPayload(req, res);
            logReq(req, status, t.ms());
        });
        s.Put(R"(/v1/vectors/(\d+)/payload)",
              [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleSetPayload(req, res);
            logReq(req, status, t.ms());
        });
        s.Patch(R"(/v1/vectors/(\d+)/payload)",
                [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handlePatchPayload(req, res);
            logReq(req, status, t.ms());
        });

        // Search.
        s.Post("/v1/search",
               [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleSearch(req, res);
            logReq(req, status, t.ms());
        });

        // Bulk build (initial-load only; binary octet-stream body).
        s.Post("/v1/build/bulk",
               [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = handleBulkBuild(req, res);
            logReq(req, status, t.ms());
        });
    }

private:
    // Live DB pointer (null when uninitialized). Copy under the lock so the
    // DB stays alive for the caller even if a concurrent DELETE swaps it out.
    std::shared_ptr<LSMVecDB> dbShared() {
        std::lock_guard<std::mutex> lk(st_->mu);
        return st_->db;
    }

    // Data-route guard: returns the live DB, or null after sending a 409.
    std::shared_ptr<LSMVecDB> requireInit(httplib::Response& res) {
        auto sp = dbShared();
        if (!sp) {
            sendError(res, 409, "index_not_initialized",
                      "index has no dimension yet; call PUT /v1/index first");
        }
        return sp;
    }

    int statusFromDb(const Status& s) {
        if (s.ok()) return 200;
        if (s.IsNotFound()) return 404;
        if (s.IsInvalidArgument()) return 400;
        if (s.IsNotSupported()) return 400;
        return 503;
    }

    int handleInsert(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what());
            return 400;
        }
        if (!body.contains("id") || !body.contains("vector")) {
            sendError(res, 400, "missing_field", "id and vector are required");
            return 400;
        }
        node_id_t id;
        try {
            if (body["id"].is_string()) {
                std::string err;
                if (!parseId(body["id"].get<std::string>(), &id, &err)) {
                    sendError(res, 400, "bad_id", err); return 400;
                }
            } else if (body["id"].is_number_unsigned()) {
                id = static_cast<node_id_t>(body["id"].get<unsigned long long>());
            } else {
                // Reject floats / negatives / non-numbers (no silent truncation).
                sendError(res, 400, "bad_id",
                          "id must be a non-negative integer or its string form");
                return 400;
            }
        } catch (...) {
            sendError(res, 400, "bad_id", "id must be a u64 or its string form");
            return 400;
        }

        std::vector<float> v;
        std::string err;
        if (!extractVector(body["vector"], &v, &err)) {
            sendError(res, 400, "bad_vector", err);
            return 400;
        }
        if (static_cast<int>(v.size()) != cfg_.dim) {
            sendError(res, 400, "wrong_dim",
                      "vector dim is " + std::to_string(v.size()) +
                      " but DB dim is " + std::to_string(cfg_.dim));
            return 400;
        }

        Status s;
        if (body.contains("metadata")) {
            std::string md = body["metadata"].is_string()
                ? body["metadata"].get<std::string>()
                : body["metadata"].dump();
            s = db_->Insert(id, Span<float>(v), std::string_view(md));
        } else {
            s = db_->Insert(id, Span<float>(v));
        }
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        sendJson(res, 201, {{"id", id}});
        return 201;
    }

    int handleInsertBatch(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        static constexpr std::size_t kMaxBatchItems = 10000;
        static constexpr std::size_t kMaxBatchBytes = 64 * 1024 * 1024;  // 64 MB
        // R3: bound the body BEFORE parsing (global cap is bulk_build_max_length).
        if (req.body.size() > kMaxBatchBytes) {
            sendError(res, 413, "batch_too_large",
                      "body exceeds " + std::to_string(kMaxBatchBytes) + " bytes");
            return 413;
        }
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what());
            return 400;
        }
        if (!body.contains("items") || !body["items"].is_array()) {
            sendError(res, 400, "missing_field", "items array is required");
            return 400;
        }
        const auto& items = body["items"];
        if (items.empty()) {
            sendError(res, 400, "empty_batch", "items must be non-empty");
            return 400;
        }
        if (items.size() > kMaxBatchItems) {
            sendError(res, 413, "batch_too_large",
                      "batch has " + std::to_string(items.size()) +
                      " items, max is " + std::to_string(kMaxBatchItems));
            return 413;
        }

        // Preflight: parse + FULLY validate every item (id range, vector, dim,
        // metadata) before any write — so a malformed item writes nothing.
        const std::size_t n = items.size();
        std::vector<node_id_t> ids(n);
        std::vector<std::vector<float>> vecs(n);
        std::vector<std::string> metas(n);
        std::vector<char> has_meta(n, 0);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& it = items[k];
            const std::string at = "item " + std::to_string(k) + ": ";
            if (!it.contains("id") || !it.contains("vector")) {
                sendError(res, 400, "missing_field", at + "id and vector are required");
                return 400;
            }
            try {
                if (it["id"].is_string()) {
                    std::string err;
                    if (!parseId(it["id"].get<std::string>(), &ids[k], &err)) {
                        sendError(res, 400, "bad_id", at + err);
                        return 400;
                    }
                } else if (it["id"].is_number_unsigned()) {
                    ids[k] = static_cast<node_id_t>(it["id"].get<unsigned long long>());
                } else {
                    // Reject floats / negatives / non-numbers (no silent truncation).
                    sendError(res, 400, "bad_id",
                              at + "id must be a non-negative integer or its string form");
                    return 400;
                }
            } catch (...) {
                sendError(res, 400, "bad_id", at + "id must be a u64 or its string form");
                return 400;
            }
            std::string err;
            if (!extractVector(it["vector"], &vecs[k], &err)) {
                sendError(res, 400, "bad_vector", at + err);
                return 400;
            }
            if (static_cast<int>(vecs[k].size()) != cfg_.dim) {
                sendError(res, 400, "wrong_dim",
                          at + "vector dim is " + std::to_string(vecs[k].size()) +
                          " but DB dim is " + std::to_string(cfg_.dim));
                return 400;
            }
            if (it.contains("metadata")) {
                metas[k] = it["metadata"].is_string()
                    ? it["metadata"].get<std::string>()
                    : it["metadata"].dump();
                has_meta[k] = 1;
            }
            // R1: id-range + metadata validation via the engine (no write).
            Status vs = db_->ValidateInsert(
                ids[k], Span<float>(vecs[k]),
                has_meta[k] ? std::string_view(metas[k]) : std::string_view{});
            if (!vs.ok()) {
                int code = statusFromDb(vs);
                sendError(res, code, "bad_item", at + vs.ToString());
                return code;
            }
        }

        // Insert all (sequential). On a mid-batch engine error, report progress.
        for (std::size_t k = 0; k < n; ++k) {
            Status s = has_meta[k]
                ? db_->Insert(ids[k], Span<float>(vecs[k]), std::string_view(metas[k]))
                : db_->Insert(ids[k], Span<float>(vecs[k]));
            if (!s.ok()) {
                int code = statusFromDb(s);
                sendJson(res, code, {{"code", "db_error"}, {"error", s.ToString()},
                                     {"inserted", static_cast<uint64_t>(k)},
                                     {"failed_index", static_cast<uint64_t>(k)}});
                return code;
            }
        }
        sendJson(res, 201, {{"inserted", static_cast<uint64_t>(n)}});
        return 201;
    }

    int handleGet(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        node_id_t id;
        std::string err;
        if (!parseId(req.matches[1].str(), &id, &err)) {
            sendError(res, 400, "bad_id", err); return 400;
        }
        std::vector<float> v;
        auto s = db_->Get(id, &v);
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        sendJson(res, 200, {{"id", id}, {"vector", v}});
        return 200;
    }

    int handlePut(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        node_id_t id;
        std::string err;
        if (!parseId(req.matches[1].str(), &id, &err)) {
            sendError(res, 400, "bad_id", err); return 400;
        }
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what()); return 400;
        }
        if (!body.contains("vector")) {
            sendError(res, 400, "missing_field", "vector is required"); return 400;
        }
        std::vector<float> v;
        if (!extractVector(body["vector"], &v, &err)) {
            sendError(res, 400, "bad_vector", err); return 400;
        }
        if (static_cast<int>(v.size()) != cfg_.dim) {
            sendError(res, 400, "wrong_dim",
                      "vector dim is " + std::to_string(v.size()) +
                      " but DB dim is " + std::to_string(cfg_.dim));
            return 400;
        }
        auto s = db_->Update(id, Span<float>(v));
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        sendJson(res, 200, {{"id", id}});
        return 200;
    }

    int handleDelete(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        node_id_t id;
        std::string err;
        if (!parseId(req.matches[1].str(), &id, &err)) {
            sendError(res, 400, "bad_id", err); return 400;
        }
        auto s = db_->Delete(id);
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        res.status = 204;
        return 204;
    }

    int handleGetPayload(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        node_id_t id;
        std::string err;
        if (!parseId(req.matches[1].str(), &id, &err)) {
            sendError(res, 400, "bad_id", err); return 400;
        }
        std::string out;
        auto s = db_->GetPayload(id, &out);
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        // GetPayload returns a JSON string; forward as JSON body.
        res.status = 200;
        res.set_content(out, "application/json");
        return 200;
    }

    int handleSetPayload(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        node_id_t id;
        std::string err;
        if (!parseId(req.matches[1].str(), &id, &err)) {
            sendError(res, 400, "bad_id", err); return 400;
        }
        // Validate body is JSON (we forward the raw text).
        try { (void)json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what()); return 400;
        }
        auto s = db_->SetPayload(id, std::string_view(req.body));
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        sendJson(res, 200, {{"id", id}});
        return 200;
    }

    int handlePatchPayload(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        node_id_t id;
        std::string err;
        if (!parseId(req.matches[1].str(), &id, &err)) {
            sendError(res, 400, "bad_id", err); return 400;
        }
        try { (void)json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what()); return 400;
        }
        auto s = db_->UpdatePayload(id, std::string_view(req.body));
        if (!s.ok()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        sendJson(res, 200, {{"id", id}});
        return 200;
    }

    int handleSearch(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what()); return 400;
        }
        if (!body.contains("vector")) {
            sendError(res, 400, "missing_field", "vector is required"); return 400;
        }
        std::vector<float> q;
        std::string err;
        if (!extractVector(body["vector"], &q, &err)) {
            sendError(res, 400, "bad_vector", err); return 400;
        }
        if (static_cast<int>(q.size()) != cfg_.dim) {
            sendError(res, 400, "wrong_dim",
                      "vector dim is " + std::to_string(q.size()) +
                      " but DB dim is " + std::to_string(cfg_.dim));
            return 400;
        }
        SearchOptions opts;
        opts.k = body.value("k", 10);
        opts.ef_search = body.value("ef_search", cfg_.ef_search_default);
        opts.max_scan_candidates = body.value("max_scan_candidates", 0);

        std::vector<SearchResult> results;
        Status s;
        if (body.contains("filter")) {
            std::string filter = body["filter"].is_string()
                ? body["filter"].get<std::string>()
                : body["filter"].dump();
            s = db_->SearchKnn(Span<float>(q), opts, std::string_view(filter), &results);
        } else {
            s = db_->SearchKnn(Span<float>(q), opts, &results);
        }
        if (!s.ok() && !s.IsNotFound()) {
            sendError(res, statusFromDb(s), "db_error", s.ToString());
            return statusFromDb(s);
        }
        json out = json::array();
        for (const auto& r : results) {
            out.push_back({{"id", r.id}, {"distance", r.distance}});
        }
        sendJson(res, 200, {{"results", out}});
        return 200;
    }

    int handleBulkBuild(const httplib::Request& req, httplib::Response& res) {
        auto sp = requireInit(res);
        if (!sp) return 409;
        LSMVecDB* db_ = sp.get();
        // Header-driven shape (avoids JSON parsing of a 50+ MB blob).
        auto get_hdr = [&](const char* k) -> std::string {
            auto it = req.headers.find(k);
            return it == req.headers.end() ? "" : it->second;
        };
        std::string h_n   = get_hdr("X-LSMVec-N");
        std::string h_dim = get_hdr("X-LSMVec-Dim");
        if (h_n.empty() || h_dim.empty()) {
            sendError(res, 400, "missing_header",
                      "X-LSMVec-N and X-LSMVec-Dim required");
            return 400;
        }
        long long n_ll = 0, dim_ll = 0;
        try {
            n_ll = std::stoll(h_n);
            dim_ll = std::stoll(h_dim);
        } catch (...) {
            sendError(res, 400, "bad_header",
                      "X-LSMVec-N / X-LSMVec-Dim must be integers");
            return 400;
        }
        if (n_ll <= 0 || dim_ll <= 0) {
            sendError(res, 400, "bad_header",
                      "n and dim must be positive");
            return 400;
        }
        if (dim_ll != cfg_.dim) {
            sendError(res, 400, "wrong_dim",
                      "X-LSMVec-Dim (" + std::to_string(dim_ll) +
                      ") does not match DB dim (" +
                      std::to_string(cfg_.dim) + ")");
            return 400;
        }
        const std::size_t vector_bytes =
            static_cast<std::size_t>(n_ll) *
            static_cast<std::size_t>(dim_ll) * sizeof(float);
        if (req.body.size() < vector_bytes) {
            sendError(res, 400, "bad_payload",
                      "body size " + std::to_string(req.body.size()) +
                      " < n*dim*4 = " + std::to_string(vector_bytes));
            return 400;
        }
        // Optional trailing payloads: bytes beyond the vector blob are a JSON
        // array of n payload objects (or nulls), positionally mapped to ids
        // 0..n-1. Validate SHAPE + size here (before BulkBuild) so malformed
        // payloads never build a half-index; collect (id, json) for an atomic
        // post-build write.
        static constexpr std::size_t kMaxPayloadBytes = 64 * 1024;  // matches DB cap
        std::vector<std::pair<node_id_t, std::string>> payload_items;
        if (req.body.size() > vector_bytes) {
            std::string_view tail(req.body.data() + vector_bytes,
                                  req.body.size() - vector_bytes);
            json payloads;
            try { payloads = json::parse(tail); }
            catch (const std::exception& e) {
                sendError(res, 400, "bad_payloads",
                          std::string("payload tail JSON parse error: ") + e.what());
                return 400;
            }
            if (!payloads.is_array() ||
                payloads.size() != static_cast<std::size_t>(n_ll)) {
                sendError(res, 400, "bad_payloads",
                          "payloads must be a JSON array of length n=" +
                          std::to_string(n_ll));
                return 400;
            }
            payload_items.reserve(payloads.size());
            for (std::size_t i = 0; i < payloads.size(); ++i) {
                if (payloads[i].is_null()) continue;
                if (!payloads[i].is_object()) {
                    sendError(res, 400, "bad_payloads",
                              "payload " + std::to_string(i) +
                              " must be an object or null");
                    return 400;
                }
                std::string md = payloads[i].dump();
                if (md.size() > kMaxPayloadBytes) {
                    sendError(res, 400, "bad_payloads",
                              "payload " + std::to_string(i) + " exceeds " +
                              std::to_string(kMaxPayloadBytes) + " bytes");
                    return 400;
                }
                payload_items.emplace_back(static_cast<node_id_t>(i),
                                           std::move(md));
            }
        }

        BulkBuildOptions bopts;
        std::string h_threads = get_hdr("X-LSMVec-Threads");
        if (!h_threads.empty()) {
            try { bopts.num_threads = std::stoi(h_threads); }
            catch (...) {}
        }
        // Default to single-threaded build to match the server's
        // single-thread-per-container philosophy. Caller can opt in
        // to multi-threaded by setting X-LSMVec-Threads.
        if (bopts.num_threads <= 0) bopts.num_threads = 1;

        const float* vectors =
            reinterpret_cast<const float*>(req.body.data());
        const int n = static_cast<int>(n_ll);

        auto t0 = std::chrono::high_resolution_clock::now();
        Status s = db_->BulkBuild(
            Span<const float>(vectors,
                              static_cast<std::size_t>(n) *
                                  static_cast<std::size_t>(dim_ll)),
            n, bopts);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!s.ok()) {
            int status = statusFromDb(s);
            // Map "DB already has data" InvalidArgument to a clearer code.
            std::string code = (status == 400) ? "db_not_empty" : "db_error";
            sendError(res, status, code, s.ToString());
            return status;
        }

        // Attach payloads (validated pre-build) atomically. Best-effort: if this
        // fails the index is already built (DB non-empty) and bulk_build cannot
        // retry, so the caller must rebuild — signaled by rebuild_required.
        std::size_t payloads_written = 0;
        if (!payload_items.empty()) {
            Status ps = db_->SetPayloadBatch(payload_items);
            if (!ps.ok()) {
                int status = statusFromDb(ps);
                sendJson(res, status, {{"code", "payloads_failed"},
                                       {"error", ps.ToString()},
                                       {"n", n},
                                       {"payloads_written", 0},
                                       {"rebuild_required", true}});
                return status;
            }
            payloads_written = payload_items.size();
        }

        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        double vps = static_cast<double>(n) / (elapsed_ms / 1000.0);
        sendJson(res, 200, {{"n", n},
                            {"elapsed_ms", elapsed_ms},
                            {"vectors_per_sec", vps},
                            {"threads", bopts.num_threads},
                            {"payloads_written", payloads_written}});
        return 200;
    }

    // ---- index lifecycle ----

    int handleCreateIndex(const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            sendError(res, 400, "bad_json", e.what()); return 400;
        }
        if (!body.contains("dim")) {
            sendError(res, 400, "missing_field", "dim is required"); return 400;
        }
        if (!body["dim"].is_number_unsigned()) {
            sendError(res, 400, "bad_dim", "dim must be a positive integer");
            return 400;
        }
        const unsigned long long dim_u = body["dim"].get<unsigned long long>();
        if (dim_u == 0 || dim_u > 100000) {
            sendError(res, 400, "bad_dim", "dim must be in [1, 100000]");
            return 400;
        }
        const int dim = static_cast<int>(dim_u);

        if (body.contains("metric") && !body["metric"].is_string()) {
            sendError(res, 400, "bad_metric",
                      "metric must be a string ('l2' or 'cosine')");
            return 400;
        }
        std::string metric = body.value("metric", std::string("l2"));
        for (auto& c : metric) c = static_cast<char>(std::tolower(c));
        DistanceMetric dm;
        if      (metric == "l2")     dm = DistanceMetric::kL2;
        else if (metric == "cosine") dm = DistanceMetric::kCosine;
        else {
            sendError(res, 400, "bad_metric", "metric must be 'l2' or 'cosine'");
            return 400;
        }

        std::lock_guard<std::mutex> lk(st_->mu);
        if (st_->db) {
            sendError(res, 409, "index_already_initialized",
                      "index already has dimension " + std::to_string(cfg_.dim));
            return 409;
        }

        // Persist dim/metric, then open. Use a temp config so a failed open
        // leaves cfg_ untouched (still uninitialized) and retryable.
        std::string werr;
        if (!writeBootstrap(cfg_.data_dir, {dim, metric}, &werr)) {
            sendError(res, 500, "bootstrap_write_failed", werr); return 500;
        }
        HttpServerConfig tmp = cfg_;
        tmp.dim = dim;
        tmp.metric = dm;
        std::unique_ptr<LSMVecDB> dbu;
        auto st = LSMVecDB::Open(tmp.data_dir, buildOptions(tmp), &dbu);
        if (!st.ok()) {
            std::error_code ec;
            std::filesystem::remove(
                std::filesystem::path(cfg_.data_dir) / kBootstrapFileName, ec);
            sendError(res, 500, "open_failed", st.ToString());
            return 500;
        }
        cfg_.dim = dim;
        cfg_.metric = dm;
        st_->db = std::move(dbu);
        std::cerr << "{\"event\":\"index_created\",\"dim\":" << dim
                  << ",\"metric\":\"" << metric << "\"}\n";
        sendJson(res, 200,
                 {{"dim", dim}, {"metric", metric}, {"initialized", true}});
        return 200;
    }

    int handleDeleteIndex(const httplib::Request& req, httplib::Response& res) {
        (void)req;
        namespace fs = std::filesystem;
        std::lock_guard<std::mutex> lk(st_->mu);
        // Close the DB (graceful RocksDB/Aster shutdown via destructor) BEFORE
        // touching files so handles are released.
        st_->db.reset();

        // Wipe the on-disk index: remove the CONTENTS of data_dir (bootstrap,
        // RocksGraph, vector.log, metadata/), never the mount point itself.
        // Remove the bootstrap FIRST so that even if a later removal fails, a
        // restart comes up cleanly uninitialized. This is a destructive
        // endpoint: a non-empty failure must surface as 500, never a false 200.
        fs::path dir(cfg_.data_dir);
        std::string failed;     // first path that could not be removed
        std::string err_msg;

        std::error_code bec;
        fs::remove(dir / kBootstrapFileName, bec);  // false (no ec) if absent
        if (bec) { failed = (dir / kBootstrapFileName).string(); err_msg = bec.message(); }

        if (failed.empty()) {
            std::error_code iec;
            for (fs::directory_iterator it(dir, iec), end; it != end; it.increment(iec)) {
                if (iec) { failed = dir.string(); err_msg = iec.message(); break; }
                std::error_code rec;
                fs::remove_all(it->path(), rec);
                if (rec) { failed = it->path().string(); err_msg = rec.message(); break; }
            }
        }

        // The DB is closed regardless, so in-memory state is uninitialized.
        cfg_.dim = 0;
        cfg_.metric = DistanceMetric::kL2;

        if (!failed.empty()) {
            std::cerr << "{\"event\":\"index_delete_failed\",\"path\":\"" << failed
                      << "\",\"error\":\"" << err_msg << "\"}\n";
            sendError(res, 500, "delete_failed",
                      "failed to remove '" + failed + "': " + err_msg);
            return 500;
        }
        std::cerr << "{\"event\":\"index_deleted\"}\n";
        sendJson(res, 200, {{"initialized", false}});
        return 200;
    }

    int handleGetIndex(const httplib::Request& req, httplib::Response& res) {
        (void)req;
        std::lock_guard<std::mutex> lk(st_->mu);
        json body;
        if (st_->db) {
            body = {{"initialized", true}, {"dim", cfg_.dim},
                    {"metric", metricToString(cfg_.metric)}};
        } else {
            body = {{"initialized", false}, {"dim", nullptr}, {"metric", nullptr}};
        }
        sendJson(res, 200, body);
        return 200;
    }

    ServerState*       st_;
    HttpServerConfig&  cfg_;   // alias of st_->cfg (dim/metric mutable)
};

}  // namespace

// ============================================================
// Top-level server runner
// ============================================================

int RunHttpServer(const HttpServerConfig& cfg) {
    // 0. Block SIGTERM/SIGINT in MAIN thread BEFORE any other thread
    // spawns. RocksDB / Aster spawns background threads inside
    // LSMVecDB::Open (compaction, flush). httplib spawns workers in
    // srv.listen. Both inherit the main thread's signal mask at spawn
    // time. If we block after these spawns, the background threads
    // have SIGTERM unblocked → kernel can deliver to them → default
    // action (terminate) → process dies before our handler runs.
    blockShutdownSignalsInThisThread();

    // 1. Server state. Open the DB now only if a dimension is known (existing
    //    bootstrap.json, or LSMVEC_DIM on first boot). Otherwise stay
    //    UNINITIALIZED — the DB opens later when a client calls PUT /v1/index.
    ServerState state;
    state.cfg = cfg;

    if (cfg.dim > 0) {
        std::unique_ptr<LSMVecDB> db;
        auto st = LSMVecDB::Open(cfg.data_dir, buildOptions(cfg), &db);
        if (!st.ok()) {
            std::cerr << "{\"event\":\"open_failed\",\"error\":\""
                      << st.ToString() << "\"}\n";
            return 1;
        }
        state.db = std::move(db);
        std::cerr << "{\"event\":\"db_open\",\"data_dir\":\""
                  << cfg.data_dir << "\",\"dim\":" << cfg.dim
                  << ",\"edge_cache_size\":" << cfg.edge_cache_size
                  << ",\"enable_stats\":" << (cfg.enable_stats ? "true" : "false")
                  << "}\n";
    } else {
        std::cerr << "{\"event\":\"uninitialized\",\"data_dir\":\""
                  << cfg.data_dir
                  << "\",\"note\":\"awaiting PUT /v1/index to set dimension\"}\n";
    }

    // 2. Build server, register routes.
    httplib::Server srv;
    // We need a single global payload cap on httplib (no per-route knob
    // in v0.18). Use the bulk-build ceiling so /v1/build/bulk can accept
    // datasets up to ~4 GB. The normal endpoints still enforce their
    // smaller cap inside the handler via Content-Length checks.
    srv.set_payload_max_length(cfg.bulk_build_max_length);
    srv.set_read_timeout(cfg.read_timeout_sec, 0);
    srv.set_write_timeout(cfg.write_timeout_sec, 0);

    Handlers handlers(&state);
    handlers.registerRoutes(srv);

    // Reject any unknown path with a JSON 404. ONLY fill the body when
    // no handler ran (res.body empty); otherwise the handler already
    // wrote a useful error body that we must not overwrite.
    srv.set_error_handler([](const httplib::Request& /*req*/,
                             httplib::Response& res) {
        if (!res.body.empty()) return;
        if (res.status == -1) res.status = 404;
        json body = {{"error", "not found"}, {"code", "not_found"}};
        res.set_content(body.dump(), "application/json");
    });

    if (cfg.http_threads > 0) {
        srv.new_task_queue = [n = cfg.http_threads] {
            return new httplib::ThreadPool(n);
        };
    }

    g_server_ptr.store(&srv);
    startShutdownWatcher();

    const int effective_threads =
        cfg.http_threads > 0
            ? cfg.http_threads
            : 1;  // matches the default in LoadHttpConfigFromEnv
    std::cerr << "{\"event\":\"listening\",\"port\":" << cfg.port
              << ",\"threads\":" << effective_threads
              << "}\n";

    bool ok = srv.listen("0.0.0.0", cfg.port);
    if (!ok && !g_shutting_down.load()) {
        std::cerr << "{\"event\":\"listen_failed\",\"port\":" << cfg.port
                  << "}\n";
        return 1;
    }

    // 3. Graceful shutdown: server returned (either listen failed OR
    // signal handler called stop()). Flush + close under the lock (a
    // concurrent create/delete also takes it).
    std::cerr << "{\"event\":\"shutdown_begin\"}\n";
    {
        std::lock_guard<std::mutex> lk(state.mu);
        if (state.db) {
            state.db->flushVectorWrites();
            state.db->Close();
            state.db.reset();
        }
    }
    std::cerr << "{\"event\":\"shutdown_done\"}\n";
    return 0;
}

}  // namespace lsm_vec
