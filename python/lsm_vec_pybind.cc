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
        .def_readwrite("vector_file_path", &lsm_vec::LSMVecDBOptions::vector_file_path)
        .def_readwrite("log_file_path", &lsm_vec::LSMVecDBOptions::log_file_path);

    py::class_<lsm_vec::SearchOptions>(m, "SearchOptions")
        .def(py::init<>())
        .def_readwrite("k", &lsm_vec::SearchOptions::k)
        .def_readwrite("ef_search", &lsm_vec::SearchOptions::ef_search);

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
        .def("insert", [](lsm_vec::LSMVecDB& db, int id, const py::sequence& seq) {
            auto data = ToVector(seq);
            RaiseStatus(db.Insert(id, MakeSpan(data)));
        })
        .def("insert", [](lsm_vec::LSMVecDB& db,
                           int id,
                           const py::array_t<float, py::array::c_style | py::array::forcecast>& array) {
            auto data = ToVector(array);
            RaiseStatus(db.Insert(id, MakeSpan(data)));
        })
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
        .def("close", [](lsm_vec::LSMVecDB& db) {
            RaiseStatus(db.Close());
        });
}
