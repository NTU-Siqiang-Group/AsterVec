#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "astervec_db.h"
#include "astervec_index.h"

namespace py = pybind11;

namespace {

std::vector<float> ToVector(const py::array_t<float, py::array::c_style | py::array::forcecast>& array)
{
    if (array.ndim() != 1) {
        throw py::value_error("expected a 1D float array");
    }
    std::vector<float> data(array.size());
    auto buf = array.unchecked<1>();
    for (ssize_t i = 0; i < buf.size(); ++i) {
        data[static_cast<size_t>(i)] = buf(i);
    }
    return data;
}

std::vector<float> ToVector(const py::sequence& seq)
{
    std::vector<float> data;
    data.reserve(static_cast<size_t>(py::len(seq)));
    for (auto item : seq) {
        data.push_back(py::cast<float>(item));
    }
    return data;
}

void RaiseStatus(const astervec::Status& status)
{
    if (status.ok()) {
        return;
    }

    std::string message = status.ToString();
    if (status.IsInvalidArgument()) {
        throw py::value_error(message);
    }
    if (status.IsNotFound()) {
        throw py::key_error(message);
    }
    if (status.IsNotSupported()) {
        throw py::value_error(message);
    }
    throw std::runtime_error(message);
}

astervec::Span<float> MakeSpan(std::vector<float>& data)
{
    return astervec::Span<float>(data.data(), data.size());
}

} // namespace

PYBIND11_MODULE(astervec, m)
{
    m.doc() = "AsterVec Python SDK";

    py::enum_<astervec::DistanceMetric>(m, "DistanceMetric")
        .value("L2", astervec::DistanceMetric::kL2)
        .value("Cosine", astervec::DistanceMetric::kCosine)
        .export_values();

    py::class_<astervec::AsterVecDBOptions>(m, "AsterVecDBOptions")
        .def(py::init<>())
        .def_readwrite("dim", &astervec::AsterVecDBOptions::dim)
        .def_readwrite("metric", &astervec::AsterVecDBOptions::metric)
        .def_readwrite("m", &astervec::AsterVecDBOptions::m)
        .def_readwrite("m_max", &astervec::AsterVecDBOptions::m_max)
        .def_readwrite("m_level", &astervec::AsterVecDBOptions::m_level)
        .def_readwrite("ef_construction", &astervec::AsterVecDBOptions::ef_construction)
        .def_readwrite("vec_file_capacity", &astervec::AsterVecDBOptions::vec_file_capacity)
        .def_readwrite("paged_max_cached_pages", &astervec::AsterVecDBOptions::paged_max_cached_pages)
        .def_readwrite("vector_storage_type", &astervec::AsterVecDBOptions::vector_storage_type)
        .def_readwrite("db_target_size", &astervec::AsterVecDBOptions::db_target_size)
        .def_readwrite("random_seed", &astervec::AsterVecDBOptions::random_seed)
        .def_readwrite("enable_stats", &astervec::AsterVecDBOptions::enable_stats)
        .def_readwrite("reinit", &astervec::AsterVecDBOptions::reinit)
        .def_readwrite("enable_batch_read", &astervec::AsterVecDBOptions::enable_batch_read)
        .def_readwrite("k", &astervec::AsterVecDBOptions::k)
        .def_readwrite("ef_search", &astervec::AsterVecDBOptions::ef_search)
        .def_readwrite("vector_file_path", &astervec::AsterVecDBOptions::vector_file_path)
        .def_readwrite("log_file_path", &astervec::AsterVecDBOptions::log_file_path);

    py::class_<astervec::SearchOptions>(m, "SearchOptions")
        .def(py::init<>())
        .def_readwrite("k", &astervec::SearchOptions::k)
        .def_readwrite("ef_search", &astervec::SearchOptions::ef_search)
        .def_readwrite("max_scan_candidates", &astervec::SearchOptions::max_scan_candidates);

    py::class_<astervec::SearchResult>(m, "SearchResult")
        .def_readonly("id", &astervec::SearchResult::id)
        .def_readonly("distance", &astervec::SearchResult::distance);

    py::class_<astervec::AsterVecDB, std::unique_ptr<astervec::AsterVecDB>>(m, "AsterVecDB")
        .def_static("open", [](const std::string& path, const astervec::AsterVecDBOptions& options) {
            std::unique_ptr<astervec::AsterVecDB> db;
            astervec::Status status = astervec::AsterVecDB::Open(path, options, &db);
            RaiseStatus(status);
            return db;
        })
        // The sequence (Python list) overload is registered BEFORE the numpy
        // array_t one so that passing a list never triggers numpy overload
        // resolution (which would import numpy and fail if it isn't installed).
        .def("insert",
             [](astervec::AsterVecDB& db,
                astervec::node_id_t id,
                const py::sequence& seq,
                py::object metadata) {
                 std::vector<float> data = ToVector(seq);
                 if (metadata.is_none()) {
                     RaiseStatus(db.Insert(id, MakeSpan(data)));
                     return;
                 }
                 py::object json_mod = py::module_::import("json");
                 std::string md_str = json_mod.attr("dumps")(metadata).cast<std::string>();
                 RaiseStatus(db.Insert(id, MakeSpan(data), md_str));
             },
             py::arg("id"), py::arg("vector"), py::arg("metadata") = py::none(),
             "Insert a vector with optional metadata (Python dict serialized to JSON).")
        .def("insert",
             [](astervec::AsterVecDB& db,
                astervec::node_id_t id,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                py::object metadata) {
                 std::vector<float> data = ToVector(array);
                 if (metadata.is_none()) {
                     RaiseStatus(db.Insert(id, MakeSpan(data)));
                     return;
                 }
                 py::object json_mod = py::module_::import("json");
                 std::string md_str = json_mod.attr("dumps")(metadata).cast<std::string>();
                 RaiseStatus(db.Insert(id, MakeSpan(data), md_str));
             },
             py::arg("id"), py::arg("vector"), py::arg("metadata") = py::none(),
             "Insert a vector with optional metadata (Python dict serialized to JSON).")
        .def("update", [](astervec::AsterVecDB& db, int id, const py::sequence& seq) {
            auto data = ToVector(seq);
            RaiseStatus(db.Update(id, MakeSpan(data)));
        })
        .def("update", [](astervec::AsterVecDB& db,
                           int id,
                           const py::array_t<float, py::array::c_style | py::array::forcecast>& array) {
            auto data = ToVector(array);
            RaiseStatus(db.Update(id, MakeSpan(data)));
        })
        .def("delete", [](astervec::AsterVecDB& db, int id) {
            RaiseStatus(db.Delete(id));
        })
        .def("get", [](astervec::AsterVecDB& db, int id) {
            std::vector<float> out;
            RaiseStatus(db.Get(id, &out));
            py::array_t<float> result(out.size());
            auto buf = result.mutable_unchecked<1>();
            for (size_t i = 0; i < out.size(); ++i) {
                buf(static_cast<ssize_t>(i)) = out[i];
            }
            return result;
        })
        .def("search_knn", [](astervec::AsterVecDB& db,
                              const py::sequence& seq,
                              const astervec::SearchOptions& options) {
            auto data = ToVector(seq);
            std::vector<astervec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        })
        .def("search_knn", [](astervec::AsterVecDB& db,
                              const py::sequence& seq,
                              int k,
                              int ef_search) {
            auto data = ToVector(seq);
            astervec::SearchOptions options;
            options.k = k;
            options.ef_search = ef_search;
            std::vector<astervec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        }, py::arg("query"), py::arg("k"), py::arg("ef_search"))
        .def("search_knn", [](astervec::AsterVecDB& db,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                              const astervec::SearchOptions& options) {
            auto data = ToVector(array);
            std::vector<astervec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        })
        .def("search_knn", [](astervec::AsterVecDB& db,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                              int k,
                              int ef_search) {
            auto data = ToVector(array);
            astervec::SearchOptions options;
            options.k = k;
            options.ef_search = ef_search;
            std::vector<astervec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        }, py::arg("query"), py::arg("k"), py::arg("ef_search"))
        .def("search_knn", [](astervec::AsterVecDB& db,
                              const py::sequence& seq) {
            auto data = ToVector(seq);
            std::vector<astervec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), &out));
            return out;
        })
        .def("search_knn", [](astervec::AsterVecDB& db,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& array) {
            auto data = ToVector(array);
            std::vector<astervec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), &out));
            return out;
        })
        // Sequence (Python list) overload first — see the insert() note above.
        .def("search",
             [](astervec::AsterVecDB& db,
                const py::sequence& query,
                int k, int ef_search,
                py::object filter,
                int max_scan_candidates) -> py::list {
                 std::vector<float> qdata = ToVector(query);
                 astervec::SearchOptions opts;
                 opts.k = k;
                 opts.ef_search = ef_search;
                 opts.max_scan_candidates = max_scan_candidates;
                 std::vector<astervec::SearchResult> out;
                 if (filter.is_none()) {
                     RaiseStatus(db.SearchKnn(MakeSpan(qdata), opts, &out));
                 } else {
                     py::object json_mod = py::module_::import("json");
                     std::string filter_json = json_mod.attr("dumps")(filter).cast<std::string>();
                     RaiseStatus(db.SearchKnn(MakeSpan(qdata), opts, filter_json, &out));
                 }
                 py::list result;
                 for (const auto& r : out) {
                     py::dict d;
                     d["id"] = r.id;
                     d["distance"] = r.distance;
                     result.append(d);
                 }
                 return result;
             },
             py::arg("query"),
             py::arg("k") = 10,
             py::arg("ef_search") = 128,
             py::arg("filter") = py::none(),
             py::arg("max_scan_candidates") = 0,
             "kNN search with optional metadata filter. Returns list of dicts {id, distance}.")
        .def("search",
             [](astervec::AsterVecDB& db,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
                int k, int ef_search,
                py::object filter,
                int max_scan_candidates) -> py::list {
                 std::vector<float> qdata = ToVector(query);
                 astervec::SearchOptions opts;
                 opts.k = k;
                 opts.ef_search = ef_search;
                 opts.max_scan_candidates = max_scan_candidates;
                 std::vector<astervec::SearchResult> out;
                 if (filter.is_none()) {
                     RaiseStatus(db.SearchKnn(MakeSpan(qdata), opts, &out));
                 } else {
                     py::object json_mod = py::module_::import("json");
                     std::string filter_json = json_mod.attr("dumps")(filter).cast<std::string>();
                     RaiseStatus(db.SearchKnn(MakeSpan(qdata), opts, filter_json, &out));
                 }
                 py::list result;
                 for (const auto& r : out) {
                     py::dict d;
                     d["id"] = r.id;
                     d["distance"] = r.distance;
                     result.append(d);
                 }
                 return result;
             },
             py::arg("query"),
             py::arg("k") = 10,
             py::arg("ef_search") = 128,
             py::arg("filter") = py::none(),
             py::arg("max_scan_candidates") = 0,
             "kNN search with optional metadata filter. Returns list of dicts {id, distance}.")
        .def("set_payload",
             [](astervec::AsterVecDB& db, astervec::node_id_t id, py::object metadata) {
                 py::object json_mod = py::module_::import("json");
                 std::string s = json_mod.attr("dumps")(metadata).cast<std::string>();
                 RaiseStatus(db.SetPayload(id, s));
             },
             py::arg("id"), py::arg("metadata"),
             "Replace the metadata associated with id.")
        .def("update_payload",
             [](astervec::AsterVecDB& db, astervec::node_id_t id, py::object partial) {
                 py::object json_mod = py::module_::import("json");
                 std::string s = json_mod.attr("dumps")(partial).cast<std::string>();
                 RaiseStatus(db.UpdatePayload(id, s));
             },
             py::arg("id"), py::arg("metadata"),
             "Merge-patch (RFC 7396) metadata for id; None values delete fields.")
        .def("delete_payload_keys",
             [](astervec::AsterVecDB& db, astervec::node_id_t id,
                const std::vector<std::string>& keys) {
                 astervec::Span<const std::string> span(keys);
                 RaiseStatus(db.DeletePayloadKeys(id, span));
             },
             py::arg("id"), py::arg("keys"),
             "Remove the specified keys from the metadata object for id.")
        .def("get_payload",
             [](astervec::AsterVecDB& db, astervec::node_id_t id) -> py::object {
                 std::string s;
                 RaiseStatus(db.GetPayload(id, &s));
                 py::object json_mod = py::module_::import("json");
                 return json_mod.attr("loads")(s);
             },
             py::arg("id"),
             "Return the metadata dict for id (empty dict if no metadata row).")
        .def("flush_vector_writes",
             [](astervec::AsterVecDB& db) { db.flushVectorWrites(); },
             "Flush any pending vector writes to the on-disk vector storage.")
        .def("bulk_build",
             [](astervec::AsterVecDB& db,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                int threads) -> py::dict {
                 if (array.ndim() != 2) {
                     throw py::value_error("vectors must be a 2D array (n, dim)");
                 }
                 const ssize_t n = array.shape(0);
                 const ssize_t dim = array.shape(1);
                 if (n <= 0 || dim <= 0) {
                     throw py::value_error("vectors must be a non-empty (n, dim) array");
                 }
                 astervec::BulkBuildOptions bopts;
                 if (threads > 0) bopts.num_threads = threads;  // 0 -> engine auto

                 // c_style|forcecast guarantees a C-contiguous float32 buffer.
                 const float* ptr = array.data();
                 auto t0 = std::chrono::high_resolution_clock::now();
                 RaiseStatus(db.BulkBuild(
                     astervec::Span<const float>(
                         ptr, static_cast<size_t>(n) * static_cast<size_t>(dim)),
                     static_cast<int>(n), bopts));
                 auto t1 = std::chrono::high_resolution_clock::now();

                 double elapsed_ms =
                     std::chrono::duration<double, std::milli>(t1 - t0).count();
                 py::dict report;
                 report["n"] = static_cast<int>(n);
                 report["elapsed_ms"] = elapsed_ms;
                 report["vectors_per_sec"] =
                     elapsed_ms > 0.0
                         ? static_cast<double>(n) / (elapsed_ms / 1000.0)
                         : 0.0;
                 report["threads"] = bopts.num_threads;
                 return report;
             },
             py::arg("vectors"), py::arg("threads") = 0,
             "Build the whole index in one pass from a 2D (n, dim) float32 array "
             "(or a list of equal-length rows). Initial-load only: the DB must be "
             "empty; ids are assigned 0..n-1. Returns a report dict "
             "{n, elapsed_ms, vectors_per_sec, threads}.")
        .def("trim_memory",
             [](astervec::AsterVecDB& db) { db.trimMemory(); },
             "Ask the allocator to return idle heap memory to the OS "
             "(glibc malloc_trim; no-op elsewhere).")
        .def("delete_stats",
             [](const astervec::AsterVecDB& db) -> py::dict {
                 astervec::AsterVecDB::DeleteStats s = db.GetDeleteStats();
                 py::dict d;
                 d["tombstones"]          = s.tombstones;
                 d["updated_real_ids"]    = s.updated_real_ids;
                 d["total_inserts_ever"]  = s.total_inserts_ever;
                 d["tombstone_ratio"]     = s.tombstone_ratio;
                 d["bloom_capacity"]      = s.bloom_capacity;
                 d["bloom_fill_ratio"]    = s.bloom_fill_ratio;
                 d["bloom_rebuild_count"] = s.bloom_rebuild_count;
                 return d;
             },
             "Snapshot of delete/tombstone observability counters as a dict.")
        .def("close", [](astervec::AsterVecDB& db) {
            RaiseStatus(db.Close());
        });
}
