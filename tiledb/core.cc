#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#undef NPY_NO_DEPRECATED_API

#define TPY_ERROR_LOC(m) throw TileDBPyError(std::string(m) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");

#include <tiledb/tiledb> // C++

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tiledb/tiledb.h> // C

//#include "readquery.h"

/*
query requires
- ctx
- array
- query
- attribute names
- ranges or subarray

- (possibly predicted shape for MR indexing of dense?
   not sure if size estimation works -- it should)
*/


namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;

class TileDBPyError : std::runtime_error {
public:
    explicit TileDBPyError(const char * m) : std::runtime_error(m) {}
    explicit TileDBPyError(std::string m) : std::runtime_error(m.c_str()) {}

public:
    virtual const char * what() const noexcept override {return std::runtime_error::what();}

};

struct BufferInfo {

    BufferInfo(std::string name, tiledb_datatype_t type, size_t elem_bytes, size_t offsets_num) :
        name(name), type(type)
    {
        py::dtype dtype = tiledb_dtype(type);
        data = py::array(dtype, elem_bytes / dtype.itemsize());
    }

    string name;
    tiledb_datatype_t type;
    py::array data;
    vector<uint32_t> offsets;
};

py::dtype tiledb_dtype(tiledb_datatype_t type) {
    switch (type) {
        case TILEDB_INT32:
            return py::dtype("int32");
        case TILEDB_INT64:
            return py::dtype("int64");
        case TILEDB_FLOAT32:
            return py::dtype("float32");
        case TILEDB_FLOAT64:
            return py::dtype("float64");
        case TILEDB_INT8:
            return py::dtype("int8");
        case TILEDB_UINT8:
            return py::dtype("uint8");
        case TILEDB_INT16:
            return py::dtype("int16");
        case TILEDB_UINT16:
            return py::dtype("uint16");
        case TILEDB_UINT32:
            return py::dtype("uint32");
        case TILEDB_UINT64:
            return py::dtype("uint64");
        case TILEDB_STRING_ASCII:
            return py::dtype("S1");
        case TILEDB_STRING_UTF8:
            return py::dtype("U1");
        case TILEDB_STRING_UTF16:
        case TILEDB_STRING_UTF32:
            TPY_ERROR_LOC("Unimplemented UTF16 or UTF32 string conversion!");
        case TILEDB_STRING_UCS2:
        case TILEDB_STRING_UCS4:
            TPY_ERROR_LOC("Unimplemented UCS2 or UCS4 string conversion!");
        case TILEDB_CHAR:
            return py::dtype("S1");
        case TILEDB_ANY:
            TPY_ERROR_LOC("Unimplemented TILEDB_ANY conversion!"); // <TODO>
        case TILEDB_DATETIME_YEAR:
        case TILEDB_DATETIME_MONTH:
        case TILEDB_DATETIME_WEEK:
        case TILEDB_DATETIME_DAY:
        case TILEDB_DATETIME_HR:
        case TILEDB_DATETIME_MIN:
        case TILEDB_DATETIME_SEC:
        case TILEDB_DATETIME_MS:
        case TILEDB_DATETIME_US:
        case TILEDB_DATETIME_NS:
        case TILEDB_DATETIME_PS:
        case TILEDB_DATETIME_FS:
        case TILEDB_DATETIME_AS:
            TPY_ERROR_LOC("Unimplemented datetime conversion!"); // <TODO>
    }
}

class PyQuery {

private:
        tiledb_ctx_t* c_ctx_;
        tiledb_array_t* c_array_;
        Context ctx_;
        shared_ptr<tiledb::Array> array_;
        shared_ptr<tiledb::Query> query_;
        std::vector<std::string> attrs;
        map<string, BufferInfo> buffers_;
        bool include_coords_;

public:
    PyQuery() = delete;

    PyQuery(
        py::object ctx,
        py::object array,
        py::tuple attrs,
        bool include_coords) {

        tiledb_ctx_t* c_ctx_ = (py::capsule)ctx.attr("__capsule__")();
        if (c_ctx_ == nullptr)
            TPY_ERROR_LOC("Invalid context pointer!")
        ctx_ = Context(c_ctx_, false);

        tiledb_array_t* c_array_ = (py::capsule)array.attr("__capsule__")();

        /* TBD whether we use the C++ API ... */
        // we never own this pointer, pass own=false
        array_ = std::shared_ptr<tiledb::Array>(new Array(ctx_, c_array_, false),
                    [](Array* p){} /* no deleter*/);

        query_ = std::shared_ptr<tiledb::Query>(new Query(ctx_, *array_, TILEDB_READ));//,
//                     [](Query* p){} /* no deleter*/);

        include_coords_ = include_coords;
    }

    void add_dim_range(uint32_t dim_idx, py::tuple r) {
        if (py::len(r) == 0)
            return;
        else if (py::len(r) != 2)
            TPY_ERROR_LOC("Unexpected range len != 2");

        auto r0 = r[0];
        auto r1 = r[1];
        // no type-check here, because we might allow cast-conversion
        //if (r0.get_type() != r1.get_type())
        //    TPY_ERROR_LOC("Mismatched type");

        auto domain = array_->schema().domain();
        auto dim = domain.dimension(dim_idx);

        auto tiledb_type = dim.type();

        /*
        if (tiledb::impl::tiledb_string_type(tiledb_type)) {
            auto r0_str = r0.cast<std::string>();
            auto r1_str = r1.cast<std::string>();
            query_->add_range(dim_idx, r0_str, r1_str);
        } else {
            convert_type(r0, dim_data_start, tiledb_type);
            convert_type(r1, dim_data_start, tiledb_type);

            query_->add_range(dim_idx, dim_data_start, dim_data_end);
        }
        */
        try {
        switch (tiledb_type) {
            case TILEDB_INT32:
                {
                using T = int32_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_INT64:
                {
                using T = int64_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_INT8:
                {
                using T = uint8_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_UINT8:
                {
                using T = uint8_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_INT16:
                {
                using T = int16_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_UINT16: {
                using T = uint16_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_UINT32:
                {
                using T = uint32_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_UINT64:
                {
                using T = uint64_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_FLOAT32:
                {
                using T = float;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_FLOAT64:
                {
                using T = double;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                break;
                }
            case TILEDB_STRING_ASCII:
            case TILEDB_STRING_UTF8:
            case TILEDB_CHAR:
                {
                if (!py::isinstance<py::str>(r0))
                    TPY_ERROR_LOC("internal error: expected string type for var-length dim!");
                query_->add_range(dim_idx, r0.cast<string>(), r1.cast<string>());
                break;
                }
            case TILEDB_DATETIME_YEAR:
            case TILEDB_DATETIME_MONTH:
            case TILEDB_DATETIME_WEEK:
            case TILEDB_DATETIME_DAY:
            case TILEDB_DATETIME_HR:
            case TILEDB_DATETIME_MIN:
            case TILEDB_DATETIME_SEC:
            case TILEDB_DATETIME_MS:
            case TILEDB_DATETIME_US:
            case TILEDB_DATETIME_NS:
            case TILEDB_DATETIME_PS:
            case TILEDB_DATETIME_FS:
            case TILEDB_DATETIME_AS:
                TPY_ERROR_LOC("<TODO> datetime conversion unimplemented");
            default:
                TPY_ERROR_LOC("Unknown dim type conversion!");

        }
        } catch (py::cast_error &e) {
            std::string msg = "Failed to cast dim range '" + (string)py::repr(r)
                              + "' to dim type " + tiledb::impl::type_to_str(tiledb_type);
            TPY_ERROR_LOC(msg);
        }
    }

    void set_ranges(py::iterable ranges) {
        // ranges are specified as one iterable per dimension

        uint32_t dim_idx = 0;
        for (auto dim_range : ranges) {
            py::tuple dim_range_iter = dim_range.cast<py::iterable>();
            for (auto r : dim_range_iter) {
                py::tuple r_tuple = r.cast<py::tuple>();
                add_dim_range(dim_idx, r_tuple);
            }
            dim_idx++;
        }

    }

    void set_attr_buffer(std::string name, py::object data)
    {}

    bool is_var(std::string name) {
        auto domain = array_->schema().domain();
        if (domain.has_dimension(name)) {
            auto dim = domain.dimension(name);
            return dim.cell_val_num() == TILEDB_VAR_NUM;
        } else if (array_->schema().has_attribute(name)) {
            auto attr = array_->schema().attribute(name);
            return attr.cell_val_num() == TILEDB_VAR_NUM;
        } else {
            TPY_ERROR_LOC("Unknown buffer type for is_var check (expected attribute or dimension)")
        }
    }

    void set_buffer(py::str name, py::object data) {
        // set input data for an attribute or dimension buffer
        if (array_->schema().domain().has_dimension(name))
            set_buffer(name, data);
        else if (array_->schema().has_attribute(name))
            set_buffer(name, data);
        else
            TPY_ERROR_LOC("Unknown attr or dim '" + (string)name +"'")
    }

    void alloc_buffer(std::string name, tiledb_datatype_t type) {
        uint64_t buf_bytes = 0;
        uint64_t offsets_num = 0;
        if (is_var(name)) {
            auto size_pair = query_->est_result_size_var(name);
            buf_bytes = size_pair.second;
            offsets_num = size_pair.first;
        } else {
            buf_bytes = query_->est_result_size(name);
        }
        buffers_.insert(
            {name, BufferInfo(name, type, buf_bytes, offsets_num)}
        );
    }

    void submit_read() {
        auto schema = array_->schema();
        auto issparse = schema.array_type() == TILEDB_SPARSE;
        auto need_dim_buffers = include_coords_ || issparse;

        if (need_dim_buffers) {
            auto domain = schema.domain();
            for (auto dim : domain.dimensions()) {
                alloc_buffer(dim.name(), dim.type());
            }
        }

        for (auto attr_pair : schema.attributes()) {
            alloc_buffer(attr_pair.first, attr_pair.second.type());
        }

        query_->submit();
    }

    void submit_write() {}

    void submit() {
        if (array_->query_type() == TILEDB_READ)
            submit_read();
        else if (array_->query_type() == TILEDB_WRITE)
            submit_write();
         else
            TPY_ERROR_LOC("Unknown query type!")
    }

    py::array test_array() {
        py::array_t<uint8_t> a;
        a.resize({10});

        a.resize({20});
        return std::move(a);
    }
};


PYBIND11_MODULE(core, m) {
    py::class_<PyQuery>(m, "PyQuery")
        .def(py::init<py::object, py::object, py::tuple, bool>())
        .def("set_ranges", &PyQuery::set_ranges)
        .def("set_buffer", &PyQuery::set_buffer)
        .def("submit", &PyQuery::submit)
        .def("test_array", &PyQuery::test_array)
        .def("test_err", [](py::object self, std::string s) { throw TileDBPyError(s);} );

    /*
       We need to make sure C++ TileDBError is translated to a correctly-typed py error.
       Note that using py::exception(..., "TileDBError") creates a new exception
       in the *readquery* module, so we must import to reference.
    */
    static auto tiledb_py_error = (py::object) py::module::import("tiledb").attr("TileDBError");

    py::register_exception_translator([](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const TileDBPyError &e) {
                // TODO: set C++ line number if possible
                PyErr_SetString(tiledb_py_error.ptr(), e.what());
            } catch (const tiledb::TileDBError &e) {
                // TODO: set C++ line number if possible
                PyErr_SetString(tiledb_py_error.ptr(), e.what());
            }
            catch (std::exception &e) {
                std::cout << "got some other error" << e.what();
            }
        });
}

}; // namespace tiledbpy
