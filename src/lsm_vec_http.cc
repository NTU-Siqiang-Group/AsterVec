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
#include <fstream>
#include <iostream>
#include <memory>
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
        // First boot: dim must be set in env.
        if (cfg->dim <= 0) {
            *err = "LSMVEC_DIM is required on first boot (no existing DB at " +
                   cfg->data_dir + ")";
            return false;
        }
        std::string err_w;
        if (!writeBootstrap(cfg->data_dir, {cfg->dim, metric}, &err_w)) {
            *err = err_w;
            return false;
        }
    }
    return true;
}

// ============================================================
// Endpoint handlers
// ============================================================

namespace {

class Handlers {
public:
    Handlers(LSMVecDB* db, const HttpServerConfig& cfg) : db_(db), cfg_(cfg) {}

    void registerRoutes(httplib::Server& s) {
        // Health / readiness.
        s.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            sendJson(res, 200, {{"status", "ok"}});
            logReq(req, 200, t.ms());
        });
        s.Get("/ready", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            int status = (db_ != nullptr) ? 200 : 503;
            sendJson(res, status, {{"status", db_ ? "ok" : "no db"}});
            logReq(req, status, t.ms());
        });

        // Stats.
        s.Get("/v1/stats", [this](const httplib::Request& req, httplib::Response& res) {
            ReqTimer t;
            auto ds = db_->GetDeleteStats();
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
    int statusFromDb(const Status& s) {
        if (s.ok()) return 200;
        if (s.IsNotFound()) return 404;
        if (s.IsInvalidArgument()) return 400;
        if (s.IsNotSupported()) return 400;
        return 503;
    }

    int handleInsert(const httplib::Request& req, httplib::Response& res) {
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
            } else {
                id = static_cast<node_id_t>(body["id"].get<unsigned long long>());
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

    int handleGet(const httplib::Request& req, httplib::Response& res) {
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
        const std::size_t expected_bytes =
            static_cast<std::size_t>(n_ll) *
            static_cast<std::size_t>(dim_ll) * sizeof(float);
        if (req.body.size() != expected_bytes) {
            sendError(res, 400, "bad_payload",
                      "body size " + std::to_string(req.body.size()) +
                      " != n*dim*4 = " + std::to_string(expected_bytes));
            return 400;
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

        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        double vps = static_cast<double>(n) / (elapsed_ms / 1000.0);
        sendJson(res, 200, {{"n", n},
                            {"elapsed_ms", elapsed_ms},
                            {"vectors_per_sec", vps},
                            {"threads", bopts.num_threads}});
        return 200;
    }

    LSMVecDB*               db_;
    const HttpServerConfig& cfg_;
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

    // 1. Open DB.
    LSMVecDBOptions opts;
    opts.dim = cfg.dim;
    opts.metric = cfg.metric;
    opts.m = cfg.m;
    opts.m_max = cfg.m_max;
    opts.ef_construction = static_cast<float>(cfg.ef_construction);
    opts.ef_search = cfg.ef_search_default;
    opts.vec_file_capacity = cfg.vec_file_capacity;
    opts.paged_max_cached_pages = cfg.paged_max_cached_pages;
    opts.vector_storage_type = 1;
    opts.edge_cache_size = cfg.edge_cache_size;
    opts.enable_stats = cfg.enable_stats;
    opts.enable_batch_read = true;
    opts.reinit = false;
    opts.vector_file_path = cfg.data_dir + "/vector.log";

    std::unique_ptr<LSMVecDB> db;
    auto st = LSMVecDB::Open(cfg.data_dir, opts, &db);
    if (!st.ok()) {
        std::cerr << "{\"event\":\"open_failed\",\"error\":\""
                  << st.ToString() << "\"}\n";
        return 1;
    }
    std::cerr << "{\"event\":\"db_open\",\"data_dir\":\""
              << cfg.data_dir << "\",\"dim\":" << cfg.dim
              << ",\"edge_cache_size\":" << cfg.edge_cache_size
              << ",\"enable_stats\":" << (cfg.enable_stats ? "true" : "false")
              << "}\n";

    // 2. Build server, register routes.
    httplib::Server srv;
    // We need a single global payload cap on httplib (no per-route knob
    // in v0.18). Use the bulk-build ceiling so /v1/build/bulk can accept
    // datasets up to ~4 GB. The normal endpoints still enforce their
    // smaller cap inside the handler via Content-Length checks.
    srv.set_payload_max_length(cfg.bulk_build_max_length);
    srv.set_read_timeout(cfg.read_timeout_sec, 0);
    srv.set_write_timeout(cfg.write_timeout_sec, 0);

    Handlers handlers(db.get(), cfg);
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
    // signal handler called stop()). Flush + close.
    std::cerr << "{\"event\":\"shutdown_begin\"}\n";
    if (db) {
        db->flushVectorWrites();
        db->Close();
        db.reset();
    }
    std::cerr << "{\"event\":\"shutdown_done\"}\n";
    return 0;
}

}  // namespace lsm_vec
