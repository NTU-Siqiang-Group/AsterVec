#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsm_vec_db.h"
#include "lsm_vec_index.h"

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

void RaiseStatus(const lsm_vec::Status& status)
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

lsm_vec::Span<float> MakeSpan(std::vector<float>& data)
{
    return lsm_vec::Span<float>(data.data(), data.size());
}

} // namespace

PYBIND11_MODULE(lsm_vec, m)
{
    m.doc() = "LSM-Vec Python SDK";

    py::enum_<lsm_vec::DistanceMetric>(m, "DistanceMetric")
        .value("L2", lsm_vec::DistanceMetric::kL2)
        .value("Cosine", lsm_vec::DistanceMetric::kCosine)
        .export_values();

    py::class_<lsm_vec::LSMVecDBOptions>(m, "LSMVecDBOptions")
        .def(py::init<>())
        .def_readwrite("dim", &lsm_vec::LSMVecDBOptions::dim)
        .def_readwrite("metric", &lsm_vec::LSMVecDBOptions::metric)
        .def_readwrite("m", &lsm_vec::LSMVecDBOptions::m)
        .def_readwrite("m_max", &lsm_vec::LSMVecDBOptions::m_max)
        .def_readwrite("m_level", &lsm_vec::LSMVecDBOptions::m_level)
        .def_readwrite("ef_construction", &lsm_vec::LSMVecDBOptions::ef_construction)
        .def_readwrite("vec_file_capacity", &lsm_vec::LSMVecDBOptions::vec_file_capacity)
        .def_readwrite("paged_max_cached_pages", &lsm_vec::LSMVecDBOptions::paged_max_cached_pages)
        .def_readwrite("vector_storage_type", &lsm_vec::LSMVecDBOptions::vector_storage_type)
        .def_readwrite("db_target_size", &lsm_vec::LSMVecDBOptions::db_target_size)
        .def_readwrite("random_seed", &lsm_vec::LSMVecDBOptions::random_seed)
        .def_readwrite("enable_stats", &lsm_vec::LSMVecDBOptions::enable_stats)
        .def_readwrite("reinit", &lsm_vec::LSMVecDBOptions::reinit)
        .def_readwrite("enable_batch_read", &lsm_vec::LSMVecDBOptions::enable_batch_read)
        .def_readwrite("k", &lsm_vec::LSMVecDBOptions::k)
        .def_readwrite("ef_search", &lsm_vec::LSMVecDBOptions::ef_search)
        .def_readwrite("vector_file_path", &lsm_vec::LSMVecDBOptions::vector_file_path)
        .def_readwrite("log_file_path", &lsm_vec::LSMVecDBOptions::log_file_path);

    py::class_<lsm_vec::SearchOptions>(m, "SearchOptions")
        .def(py::init<>())
        .def_readwrite("k", &lsm_vec::SearchOptions::k)
        .def_readwrite("ef_search", &lsm_vec::SearchOptions::ef_search)
        .def_readwrite("max_scan_candidates", &lsm_vec::SearchOptions::max_scan_candidates);

    py::class_<lsm_vec::SearchResult>(m, "SearchResult")
        .def_readonly("id", &lsm_vec::SearchResult::id)
        .def_readonly("distance", &lsm_vec::SearchResult::distance);

    py::class_<lsm_vec::LSMVecDB, std::unique_ptr<lsm_vec::LSMVecDB>>(m, "LSMVecDB")
        .def_static("open", [](const std::string& path, const lsm_vec::LSMVecDBOptions& options) {
            std::unique_ptr<lsm_vec::LSMVecDB> db;
            lsm_vec::Status status = lsm_vec::LSMVecDB::Open(path, options, &db);
            RaiseStatus(status);
            return db;
        })
        .def("insert",
             [](lsm_vec::LSMVecDB& db,
                lsm_vec::node_id_t id,
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
        .def("insert",
             [](lsm_vec::LSMVecDB& db,
                lsm_vec::node_id_t id,
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
        .def("update", [](lsm_vec::LSMVecDB& db, int id, const py::sequence& seq) {
            auto data = ToVector(seq);
            RaiseStatus(db.Update(id, MakeSpan(data)));
        })
        .def("update", [](lsm_vec::LSMVecDB& db,
                           int id,
                           const py::array_t<float, py::array::c_style | py::array::forcecast>& array) {
            auto data = ToVector(array);
            RaiseStatus(db.Update(id, MakeSpan(data)));
        })
        .def("delete", [](lsm_vec::LSMVecDB& db, int id) {
            RaiseStatus(db.Delete(id));
        })
        .def("get", [](lsm_vec::LSMVecDB& db, int id) {
            std::vector<float> out;
            RaiseStatus(db.Get(id, &out));
            py::array_t<float> result(out.size());
            auto buf = result.mutable_unchecked<1>();
            for (size_t i = 0; i < out.size(); ++i) {
                buf(static_cast<ssize_t>(i)) = out[i];
            }
            return result;
        })
        .def("search_knn", [](lsm_vec::LSMVecDB& db,
                              const py::sequence& seq,
                              const lsm_vec::SearchOptions& options) {
            auto data = ToVector(seq);
            std::vector<lsm_vec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        })
        .def("search_knn", [](lsm_vec::LSMVecDB& db,
                              const py::sequence& seq,
                              int k,
                              int ef_search) {
            auto data = ToVector(seq);
            lsm_vec::SearchOptions options;
            options.k = k;
            options.ef_search = ef_search;
            std::vector<lsm_vec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        })
        .def("search_knn", [](lsm_vec::LSMVecDB& db,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                              const lsm_vec::SearchOptions& options) {
            auto data = ToVector(array);
            std::vector<lsm_vec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        })
        .def("search_knn", [](lsm_vec::LSMVecDB& db,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
                              int k,
                              int ef_search) {
            auto data = ToVector(array);
            lsm_vec::SearchOptions options;
            options.k = k;
            options.ef_search = ef_search;
            std::vector<lsm_vec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), options, &out));
            return out;
        })
        .def("search_knn", [](lsm_vec::LSMVecDB& db,
                              const py::sequence& seq) {
            auto data = ToVector(seq);
            std::vector<lsm_vec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), &out));
            return out;
        })
        .def("search_knn", [](lsm_vec::LSMVecDB& db,
                              const py::array_t<float, py::array::c_style | py::array::forcecast>& array) {
            auto data = ToVector(array);
            std::vector<lsm_vec::SearchResult> out;
            RaiseStatus(db.SearchKnn(MakeSpan(data), &out));
            return out;
        })
        .def("search",
             [](lsm_vec::LSMVecDB& db,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
                int k, int ef_search,
                py::object filter,
                int max_scan_candidates) -> py::list {
                 std::vector<float> qdata = ToVector(query);
                 lsm_vec::SearchOptions opts;
                 opts.k = k;
                 opts.ef_search = ef_search;
                 opts.max_scan_candidates = max_scan_candidates;
                 std::vector<lsm_vec::SearchResult> out;
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
             py::arg("ef_search") = 64,
             py::arg("filter") = py::none(),
             py::arg("max_scan_candidates") = 0,
             "kNN search with optional metadata filter. Returns list of dicts {id, distance}.")
        .def("search",
             [](lsm_vec::LSMVecDB& db,
                const py::sequence& query,
                int k, int ef_search,
                py::object filter,
                int max_scan_candidates) -> py::list {
                 std::vector<float> qdata = ToVector(query);
                 lsm_vec::SearchOptions opts;
                 opts.k = k;
                 opts.ef_search = ef_search;
                 opts.max_scan_candidates = max_scan_candidates;
                 std::vector<lsm_vec::SearchResult> out;
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
             py::arg("ef_search") = 64,
             py::arg("filter") = py::none(),
             py::arg("max_scan_candidates") = 0,
             "kNN search with optional metadata filter. Returns list of dicts {id, distance}.")
        .def("set_payload",
             [](lsm_vec::LSMVecDB& db, lsm_vec::node_id_t id, py::object metadata) {
                 py::object json_mod = py::module_::import("json");
                 std::string s = json_mod.attr("dumps")(metadata).cast<std::string>();
                 RaiseStatus(db.SetPayload(id, s));
             },
             py::arg("id"), py::arg("metadata"),
             "Replace the metadata associated with id.")
        .def("update_payload",
             [](lsm_vec::LSMVecDB& db, lsm_vec::node_id_t id, py::object partial) {
                 py::object json_mod = py::module_::import("json");
                 std::string s = json_mod.attr("dumps")(partial).cast<std::string>();
                 RaiseStatus(db.UpdatePayload(id, s));
             },
             py::arg("id"), py::arg("metadata"),
             "Merge-patch (RFC 7396) metadata for id; None values delete fields.")
        .def("delete_payload_keys",
             [](lsm_vec::LSMVecDB& db, lsm_vec::node_id_t id,
                const std::vector<std::string>& keys) {
                 lsm_vec::Span<const std::string> span(keys);
                 RaiseStatus(db.DeletePayloadKeys(id, span));
             },
             py::arg("id"), py::arg("keys"),
             "Remove the specified keys from the metadata object for id.")
        .def("get_payload",
             [](lsm_vec::LSMVecDB& db, lsm_vec::node_id_t id) -> py::object {
                 std::string s;
                 RaiseStatus(db.GetPayload(id, &s));
                 py::object json_mod = py::module_::import("json");
                 return json_mod.attr("loads")(s);
             },
             py::arg("id"),
             "Return the metadata dict for id (empty dict if no metadata row).")
        .def("flush_vector_writes",
             [](lsm_vec::LSMVecDB& db) { db.flushVectorWrites(); },
             "Flush any pending vector writes to the on-disk vector storage.")
        .def("close", [](lsm_vec::LSMVecDB& db) {
            RaiseStatus(db.Close());
        });
}
