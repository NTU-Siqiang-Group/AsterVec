// LSM-Vec HTTP/REST server.
//
// One-binary, single-tenant HTTP server. Used inside a per-user
// container at trial deployment; see the design docs §5.
//
// Configuration from environment variables (see http_main.cc).
// REST API surface in lsm_vec_http.cc.
//
// All endpoints serve directly from LSMVecDB; Phase A made the
// underlying engine thread-safe for HTTP-pool concurrency, so the
// handlers do not take any global mutex on the read path.

#pragma once

#include "lsm_vec_db.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace lsm_vec {

struct HttpServerConfig {
    // ---- HTTP listener ----
    int port = 8000;
    // Server defaults to single-threaded request handling: one user
    // per container, no internal contention. Set LSMVEC_HTTP_THREADS=N
    // to enable parallel request handling on dedicated hardware.
    int http_threads = 1;
    int read_timeout_sec = 30;
    int write_timeout_sec = 30;
    std::size_t payload_max_length = 16 * 1024 * 1024;   // 16 MB (normal endpoints)
    std::size_t bulk_build_max_length = 4ULL * 1024 * 1024 * 1024;  // 4 GB ceiling for /v1/build/bulk

    // ---- LSM-Vec engine ----
    std::string data_dir = "/data";
    int dim = 0;                     // required on first boot
    DistanceMetric metric = DistanceMetric::kL2;
    int m = 8;
    int m_max = 24;
    int ef_construction = 32;
    int ef_search_default = 128;
    std::size_t edge_cache_size = 100000;
    std::size_t vec_file_capacity = 1000000;   // initial idToPage_ size; auto-expands
    std::size_t paged_max_cached_pages = 8192;

    // ---- Observability ----
    bool enable_stats = false;       // flip to true via env for live debugging
    std::string log_level = "info";  // "trace"|"debug"|"info"|"warn"|"error"
};

// Parse env vars into config. Returns false and fills *err on bad input.
// First-boot vs reopen: dim/metric are persisted in the data dir; on
// reopen they are validated against the env values.
bool LoadHttpConfigFromEnv(HttpServerConfig* cfg, std::string* err);

// Run the HTTP server until SIGTERM/SIGINT. Returns 0 on clean exit,
// non-zero on startup or fatal error. Calls flushVectorWrites + Close
// on the underlying DB during graceful shutdown.
int RunHttpServer(const HttpServerConfig& cfg);

}  // namespace lsm_vec
