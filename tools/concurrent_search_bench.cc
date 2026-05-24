// Concurrent search throughput benchmark.
//
// Builds a LSM-Vec DB from .fvecs, then runs N-thread concurrent
// search at each requested thread count against the in-memory index
// (same process — avoids reopen quirks).
//
// Usage:
//   ./build/bin/concurrent_search_bench \
//       --db /tmp/run_100k/db \
//       --input data/sift_100k_input.fvecs \
//       --queries data/sift_100k_query.fvecs \
//       --ground data/sift_100k_groundtruth.ivecs \
//       --threads 1,2,4,8 --k 10 --efs 128

#include "lsm_vec_db.h"
#include "utils.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::string db_path;
    std::string input_file;
    std::string query_file;
    std::string ground_file;
    std::vector<int> threads_list = {1, 2, 4, 8};
    int k = 10;
    int efs = 128;
    int M = 8;
    int Mmax = 24;
    int efc = 32;
    bool reinit = true;
    size_t edge_cache_size = 100000;
    std::vector<int> build_threads_list = {1};  // sequential build by default
    bool skip_search = false;
    bool skip_build = false;
};

std::vector<int> parseThreadsList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    }
    return out;
}

Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (k == "--db")       a.db_path = next();
        else if (k == "--input")    a.input_file = next();
        else if (k == "--queries")  a.query_file = next();
        else if (k == "--ground")   a.ground_file = next();
        else if (k == "--threads")  a.threads_list = parseThreadsList(next());
        else if (k == "--k")        a.k = std::atoi(next().c_str());
        else if (k == "--efs")      a.efs = std::atoi(next().c_str());
        else if (k == "--M")        a.M = std::atoi(next().c_str());
        else if (k == "--Mmax")     a.Mmax = std::atoi(next().c_str());
        else if (k == "--efc")      a.efc = std::atoi(next().c_str());
        else if (k == "--no-reinit") a.reinit = false;
        else if (k == "--edge-cache") a.edge_cache_size = static_cast<size_t>(std::atoll(next().c_str()));
        else if (k == "--build-threads") a.build_threads_list = parseThreadsList(next());
        else if (k == "--skip-search") a.skip_search = true;
        else if (k == "--skip-build")  a.skip_build  = true;
        else { std::cerr << "Unknown flag: " << k << "\n"; std::exit(2); }
    }
    if (a.db_path.empty() || a.input_file.empty() || a.query_file.empty()) {
        std::cerr << "Required: --db <path> --input <fvecs> --queries <fvecs>\n";
        std::exit(2);
    }
    return a;
}

double recallAtK(const std::vector<std::vector<lsm_vec::SearchResult>>& results,
                 const std::vector<std::vector<int>>& ground, int k)
{
    if (ground.empty()) return -1.0;
    size_t n = std::min(results.size(), ground.size());
    if (n == 0) return -1.0;
    double total = 0.0;
    for (size_t q = 0; q < n; ++q) {
        const auto& got = results[q];
        const auto& gt  = ground[q];
        size_t limit = std::min(static_cast<size_t>(k), gt.size());
        if (limit == 0) continue;
        size_t hits = 0;
        for (size_t i = 0; i < limit; ++i) {
            int target = gt[i];
            for (const auto& r : got) {
                if (static_cast<long long>(r.id) == target) { ++hits; break; }
            }
        }
        total += static_cast<double>(hits) / static_cast<double>(limit);
    }
    return total / static_cast<double>(n);
}

}  // namespace

int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);

    // ---- Load data ----
    auto inputs = readFvecsFile(args.input_file);
    auto queries = readFvecsFile(args.query_file);
    std::vector<std::vector<int>> ground;
    if (!args.ground_file.empty()) ground = readIvecsFile(args.ground_file);
    if (inputs.empty() || queries.empty()) {
        std::cerr << "Failed to load data\n";
        return 1;
    }
    const int dim = static_cast<int>(inputs[0].size());
    std::cout << "Loaded " << inputs.size() << " inputs (dim=" << dim << "), "
              << queries.size() << " queries\n";

    std::cout << "edge_cache_size=" << args.edge_cache_size << "\n";

    auto make_opts = [&](const std::string& db_path) {
        lsm_vec::LSMVecDBOptions o;
        o.dim = dim;
        o.metric = lsm_vec::DistanceMetric::kL2;
        o.m = args.M;
        o.m_max = args.Mmax;
        o.ef_construction = static_cast<float>(args.efc);
        o.vec_file_capacity = std::max<size_t>(inputs.size() + 1024, 100000);
        o.paged_max_cached_pages = 8192;
        o.vector_storage_type = 1;
        o.k = args.k;
        o.ef_search = args.efs;
        o.enable_batch_read = true;
        o.reinit = true;
        o.edge_cache_size = args.edge_cache_size;
        o.vector_file_path = db_path + "/vector.log";
        return o;
    };

    // ---- Build (one or more thread counts, each on a fresh DB) ----
    // Last build's DB is kept open for the search benchmark below.
    std::unique_ptr<lsm_vec::LSMVecDB> db;
    if (!args.skip_build) {
        std::cout << "\n--- Build (concurrent insert on fresh DB) ---\n";
        double first_build_s = -1.0;
        for (size_t bi = 0; bi < args.build_threads_list.size(); ++bi) {
            int bt = args.build_threads_list[bi];
            if (bt < 1) continue;
            bool is_last = (bi + 1 == args.build_threads_list.size());

            std::string subdir = args.db_path + "/build_" + std::to_string(bt);
            std::error_code ec;
            std::filesystem::remove_all(subdir, ec);
            std::filesystem::create_directories(subdir, ec);

            std::unique_ptr<lsm_vec::LSMVecDB> local_db;
            auto st = lsm_vec::LSMVecDB::Open(subdir, make_opts(subdir), &local_db);
            if (!st.ok()) { std::cerr << "Open: " << st.ToString() << "\n"; return 1; }

            std::atomic<size_t> failed{0};
            auto build_worker = [&](int tid) {
                const size_t total = inputs.size();
                const size_t chunk = (total + bt - 1) / bt;
                const size_t lo = std::min(total, static_cast<size_t>(tid) * chunk);
                const size_t hi = std::min(total, lo + chunk);
                for (size_t i = lo; i < hi; ++i) {
                    auto s = local_db->Insert(
                        static_cast<uint64_t>(i),
                        lsm_vec::Span<float>(const_cast<std::vector<float>&>(inputs[i])));
                    if (!s.ok()) failed.fetch_add(1, std::memory_order_relaxed);
                }
            };

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> ts;
            ts.reserve(bt);
            for (int t = 0; t < bt; ++t) ts.emplace_back(build_worker, t);
            for (auto& t : ts) t.join();
            local_db->flushVectorWrites();
            auto t1 = std::chrono::high_resolution_clock::now();

            double build_s = std::chrono::duration<double>(t1 - t0).count();
            double qps = static_cast<double>(inputs.size()) / build_s;
            if (first_build_s < 0.0) first_build_s = build_s;
            double speedup = first_build_s / build_s;
            std::cout << "build_threads=" << bt
                      << "  elapsed=" << build_s << "s"
                      << "  insert_qps=" << qps
                      << "  speedup=" << speedup << "x"
                      << "  failed=" << failed.load() << "\n";

            if (is_last && !args.skip_search) {
                db = std::move(local_db);
            } else {
                local_db->Close();
            }
        }
    } else {
        // Open existing DB without rebuilding.
        auto o = make_opts(args.db_path);
        o.reinit = false;
        auto st = lsm_vec::LSMVecDB::Open(args.db_path, o, &db);
        if (!st.ok()) { std::cerr << "Open: " << st.ToString() << "\n"; return 1; }
    }

    if (args.skip_search || !db) return 0;

    // ---- Probe ----
    lsm_vec::SearchOptions sopts;
    sopts.k = args.k;
    sopts.ef_search = args.efs;
    {
        std::vector<lsm_vec::SearchResult> probe;
        auto s = db->SearchKnn(lsm_vec::Span<float>(queries[0]), sopts, &probe);
        std::cout << "[probe] status=" << s.ToString() << " results=" << probe.size() << "\n";
    }

    // ---- Concurrent search ----
    std::cout << "\n--- Concurrent search ---\n";
    for (int nthreads : args.threads_list) {
        if (nthreads < 1) continue;

        std::vector<std::vector<lsm_vec::SearchResult>> results(queries.size());
        std::atomic<size_t> bad{0};

        std::atomic<size_t> next_q{0};
        auto worker = [&](int /*tid*/) {
            const size_t total = queries.size();
            // Work-stealing via atomic counter to remove tail-imbalance from
            // contiguous slicing: every thread keeps pulling the next query
            // until the queue drains, so finish-times match.
            while (true) {
                size_t q = next_q.fetch_add(1, std::memory_order_relaxed);
                if (q >= total) break;
                std::vector<lsm_vec::SearchResult> out;
                auto s = db->SearchKnn(lsm_vec::Span<float>(queries[q]), sopts, &out);
                if (!s.ok()) { bad.fetch_add(1, std::memory_order_relaxed); continue; }
                results[q] = std::move(out);
            }
        };

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> ts;
        ts.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) ts.emplace_back(worker, t);
        for (auto& t : ts) t.join();
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        double qps = static_cast<double>(queries.size()) / elapsed_s;
        double recall = recallAtK(results, ground, args.k);

        std::cout << "threads=" << nthreads
                  << "  elapsed=" << elapsed_s << "s"
                  << "  qps=" << qps
                  << "  bad=" << bad.load();
        if (recall >= 0.0) std::cout << "  recall@" << args.k << "=" << recall;
        std::cout << "\n";
    }

    db->Close();
    return 0;
}
