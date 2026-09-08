// Batch2 R5: standalone driver for the cross-process NONE/VARINT
// differential. Two fresh single-threaded processes with the same
// options.random_seed build IDENTICAL graph structures (the level RNG is
// thread_local, seeded from random_seed on first use per thread), so any
// difference in printed results is attributable to the adjacency encoding
// alone. Usage: encoding_diff_driver <dbdir> <outfile>
// (encoding chosen by ASTERVEC_GRAPH_VARINT in the environment)
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "astervec_db.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <dbdir> <outfile>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string outfile = argv[2];
    const int dim = 16, n = 400;

    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    auto mkvec = [&] {
        std::vector<float> v(dim);
        for (auto& x : v) x = u(rng);
        return v;
    };
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < n; i++) vecs.push_back(mkvec());
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 25; i++) queries.push_back(mkvec());

    astervec::AsterVecDBOptions opts;
    opts.dim = dim;
    opts.m = 8;
    opts.m_max = 24;
    opts.ef_construction = 64;
    opts.vec_file_capacity = 4096;
    opts.vector_file_path = dir + "/vecs.bin";
    opts.random_seed = 424242;

    std::unique_ptr<astervec::AsterVecDB> db;
    auto st = astervec::AsterVecDB::Open(dir, opts, &db);
    if (!st.ok()) { std::fprintf(stderr, "open: %s\n", st.ToString().c_str()); return 1; }

    for (int i = 0; i < n; i++) {
        st = db->Insert(static_cast<astervec::node_id_t>(i),
                        astervec::Span<float>(vecs[i].data(), dim));
        if (!st.ok()) { std::fprintf(stderr, "insert %d: %s\n", i, st.ToString().c_str()); return 1; }
    }
    for (int i = 0; i < 30; i++) {  // updates put bit-63 ids into the graph
        st = db->Update(static_cast<astervec::node_id_t>(i),
                        astervec::Span<float>(vecs[(i + 11) % n].data(), dim));
        if (!st.ok()) { std::fprintf(stderr, "update %d: %s\n", i, st.ToString().c_str()); return 1; }
    }
    for (int i = 350; i < 370; i++) {
        st = db->Delete(static_cast<astervec::node_id_t>(i));
        if (!st.ok()) { std::fprintf(stderr, "delete %d: %s\n", i, st.ToString().c_str()); return 1; }
    }

    std::ofstream out(outfile);
    for (const auto& q : queries) {
        astervec::SearchOptions so;
        so.k = 10;
        so.ef_search = 64;
        std::vector<astervec::SearchResult> res;
        st = db->SearchKnn(astervec::Span<float>(
                               const_cast<float*>(q.data()), q.size()),
                           so, &res);
        if (!st.ok()) { std::fprintf(stderr, "search: %s\n", st.ToString().c_str()); return 1; }
        for (const auto& r : res) out << r.id << ' ';
        out << '\n';
    }
    db->Close();
    return 0;
}
