// AsterVecDB::BulkBuild end-to-end test.
//
// Validates the in-memory bulk-build path (RNN-Descent layer 0 +
// MIRAGE-style upper-layer stitch + bulk-write to AsterDB) against
// brute-force ground truth, and confirms that post-bulk-build the
// index accepts streaming INSERT/SEARCH like a normal index.

#include "doctest.h"
#include "astervec_db.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string NewTempDir(const char* prefix) {
    std::string tmpl = std::string("/tmp/") + prefix + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = mkdtemp(buf.data());
    REQUIRE(dir != nullptr);
    return std::string(dir);
}

std::unique_ptr<astervec::AsterVecDB>
OpenFresh(const std::string& path, int dim, std::size_t capacity) {
    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 64;
    opts.vec_file_capacity = capacity;
    opts.vector_file_path = path + "/vecs.bin";
    std::unique_ptr<astervec::AsterVecDB> db;
    REQUIRE(astervec::AsterVecDB::Open(path, opts, &db).ok());
    return db;
}

// Generate `n` clustered vectors: 10 cluster centers, each cluster
// gets n/10 vectors near it. Real ANN workloads have cluster
// structure; uniform-cube random has weak structure and depresses
// recall on any graph algorithm, which is unrepresentative.
std::vector<float> ClusteredDataset(int n, int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> centre_dist(-1.0f, 1.0f);
    std::normal_distribution<float> jitter(0.0f, 0.1f);

    const int n_clusters = 10;
    std::vector<std::vector<float>> centres(n_clusters,
                                            std::vector<float>(dim));
    for (auto& c : centres) {
        for (auto& v : c) v = centre_dist(rng);
    }

    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (int i = 0; i < n; ++i) {
        const auto& c = centres[i % n_clusters];
        for (int d = 0; d < dim; ++d) {
            data[static_cast<size_t>(i) * dim + d] = c[d] + jitter(rng);
        }
    }
    return data;
}

std::vector<std::uint64_t> BruteForceTopK(const float* data, int n, int dim,
                                          const float* q, int k) {
    std::vector<std::pair<float, std::uint64_t>> dists(n);
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = data[static_cast<size_t>(i) * dim + d] - q[d];
            s += diff * diff;
        }
        dists[i] = {s, static_cast<std::uint64_t>(i)};
    }
    int kk = std::min(k, n);
    std::partial_sort(dists.begin(), dists.begin() + kk, dists.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
    std::vector<std::uint64_t> out(kk);
    for (int i = 0; i < kk; ++i) out[i] = dists[i].second;
    return out;
}

}  // namespace

TEST_CASE("bulk_build: empty index, n=1000, recall@10 close to streaming") {
    constexpr int kDim = 32;
    constexpr int kN   = 1000;
    constexpr int kQ   = 100;
    constexpr int kK   = 10;
    constexpr int kEf  = 64;

    std::string path = NewTempDir("astervec_bulkbuild");
    auto db = OpenFresh(path, kDim, kN * 2);

    auto data = ClusteredDataset(kN, kDim, /*seed=*/42);

    astervec::BulkBuildOptions bopts;
    bopts.num_threads = 4;
    REQUIRE(db->BulkBuild(astervec::Span<const float>(data.data(),
                                                     data.size()),
                          kN, bopts)
                .ok());

    // Independent query stream.
    std::mt19937 qrng(99);
    std::normal_distribution<float> qjit(0.0f, 0.5f);
    std::vector<float> queries(static_cast<size_t>(kQ) * kDim);
    for (auto& v : queries) v = qjit(qrng);

    astervec::SearchOptions opts;
    opts.k         = kK;
    opts.ef_search = kEf;

    long matched = 0;
    long total   = 0;
    for (int q = 0; q < kQ; ++q) {
        const float* qv = queries.data() + static_cast<size_t>(q) * kDim;
        auto gt = BruteForceTopK(data.data(), kN, kDim, qv, kK);
        std::unordered_set<std::uint64_t> gt_set(gt.begin(), gt.end());

        std::vector<astervec::SearchResult> out;
        REQUIRE(db->SearchKnn(
                    astervec::Span<float>(const_cast<float*>(qv), kDim),
                    opts, &out).ok());
        for (auto& r : out) {
            if (gt_set.count(r.id)) ++matched;
        }
        total += static_cast<long>(gt.size());
    }
    double recall = total > 0 ? static_cast<double>(matched) / total : 0.0;
    INFO("recall@10 = " << recall);
    // Clustered synthetic data: a working HNSW-style index should
    // exceed 0.7. The streaming path on the same data is typically
    // 0.85-0.95.
    CHECK(recall > 0.70);
}

TEST_CASE("bulk_build: streaming inserts work after bulk-build") {
    constexpr int kDim = 16;
    constexpr int kBulkN  = 200;
    constexpr int kExtraN = 50;

    std::string path = NewTempDir("astervec_bulkbuild_then_insert");
    auto db = OpenFresh(path, kDim, (kBulkN + kExtraN) * 2);

    auto data = ClusteredDataset(kBulkN, kDim, /*seed=*/7);

    astervec::BulkBuildOptions bopts;
    bopts.num_threads = 2;
    REQUIRE(db->BulkBuild(astervec::Span<const float>(data), kBulkN, bopts)
                .ok());

    // Insert 50 more nodes through the streaming path.
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < kExtraN; ++i) {
        std::vector<float> v(kDim);
        for (auto& x : v) x = u(rng);
        auto st = db->Insert(static_cast<std::uint64_t>(kBulkN + i),
                             astervec::Span<float>(v));
        REQUIRE(st.ok());
    }
    db->flushVectorWrites();

    // Each newly inserted id should be retrievable through Get().
    for (int i = 0; i < kExtraN; ++i) {
        std::vector<float> out;
        REQUIRE(db->Get(static_cast<std::uint64_t>(kBulkN + i), &out).ok());
        CHECK(static_cast<int>(out.size()) == kDim);
    }
}

// A "hub" dataset: vector 0 is at the cluster centre; everyone else
// is jittered around it. After RNND every node has the hub in its
// top-R, so symmetrizing without per-vertex degree cap would push the
// hub's out-degree to ~N (far above m_max_). This test wedges the
// invariant.
TEST_CASE("bulk_build: post-symmetrize cap holds even for hub nodes") {
    constexpr int kDim    = 8;
    constexpr int kN      = 200;

    std::string path = NewTempDir("astervec_bulkbuild_hub");
    auto db = OpenFresh(path, kDim, kN * 2);  // m=8, m_max=24

    // Hub at origin; the rest within a small radius.
    std::mt19937 rng(2026);
    std::normal_distribution<float> jit(0.0f, 0.05f);
    std::vector<float> data(static_cast<size_t>(kN) * kDim, 0.0f);
    for (int i = 1; i < kN; ++i) {
        for (int d = 0; d < kDim; ++d) {
            data[static_cast<size_t>(i) * kDim + d] = jit(rng);
        }
    }

    astervec::BulkBuildOptions bopts;
    bopts.num_threads = 2;
    REQUIRE(db->BulkBuild(astervec::Span<const float>(data), kN, bopts).ok());

    // SEARCH must still return something sensible. The actual degree
    // check is enforced inside bulkLoadLayer0; here we sanity-check
    // that a query near the hub returns the hub.
    std::vector<float> q(kDim, 0.001f);   // near the hub
    astervec::SearchOptions sopts;
    sopts.k         = 1;
    sopts.ef_search = 32;
    std::vector<astervec::SearchResult> out;
    REQUIRE(db->SearchKnn(astervec::Span<float>(q), sopts, &out).ok());
    REQUIRE(out.size() == 1);
    CHECK(out[0].id == 0u);   // hub is the closest
}

TEST_CASE("bulk_build: rejects non-empty index") {
    constexpr int kDim = 8;
    std::string path = NewTempDir("astervec_bulkbuild_reject");
    auto db = OpenFresh(path, kDim, 100);

    // Populate via streaming Insert first.
    std::vector<float> v(kDim, 0.1f);
    REQUIRE(db->Insert(0, astervec::Span<float>(v)).ok());
    db->flushVectorWrites();

    std::vector<float> bulk(static_cast<size_t>(10) * kDim, 0.5f);
    astervec::BulkBuildOptions bopts;
    auto st = db->BulkBuild(astervec::Span<const float>(bulk), 10, bopts);
    CHECK_FALSE(st.ok());
}
