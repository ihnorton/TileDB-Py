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

class NPyBuffer {

    NPyBuffer(size_t bytes, size_t itemsize=1) :
        bytes_(bytes), itemsize_(itemsize), own_(true) {

        d_ = (char*)PyDataMem_NEW(bytes);
        if (d_ == nullptr)
            throw TileDBPyError(string("Failed to allocate NumPy buffer size: ") + std::to_string(bytes));

    }

    const char* data() {
        return d_;
    }

    const char* take() {
        own_ = false;
        return d_;
    }

    void realloc(size_t bytes) {
        const char* d_new = (char*)PyDataMem_RENEW(d_, bytes);
        if (d_new == nullptr)
            throw TileDBPyError(string("Failed to allocate NumPy buffer size: ") + std::to_string(bytes));
        d_ = (char*)d_new;
        bytes_ = bytes;
    }

    void resize(size_t nelem) {
        size_t new_bytes = itemsize_ * nelem;
        this->realloc(new_bytes);
    }

    void dealloc() {
        if (!own_)
            return;

        assert(d_);
        PyDataMem_FREE(d_);
    }

private:
    char* d_;
    size_t bytes_;
    size_t itemsize_;
    bool own_;
};

struct AttrInfo {
    string name;
    vector<char> data;
    vector<char> offsets;
};

class ReadQuery {
/*
    public:
        ReadQuery(
            py::object ctx_cap,
            py::object array_cap,
            py::tuple attrs,
            bool include_coords = false
        );

        void submit();
        void set_ranges(py::iterable ranges);

        void test(py::tuple x);
*/
private:
        tiledb_ctx_t* c_ctx_;
        tiledb_array_t* c_array_;
        Context ctx_;
        shared_ptr<tiledb::Array> array_;
        shared_ptr<tiledb::Query> query_;
        std::vector<std::string> attrs;
        bool include_coords_;

        // dim data scratch space
        std::vector<char> dim_data_start;
        std::vector<char> dim_data_end;

public:
    void test(py::tuple ex) {
        for (auto o : ex) {
            py::print(o);
        }
    }

    ReadQuery() = delete;

    ReadQuery(
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
        array_ = std::shared_ptr<tiledb::Array>(new Array(ctx_, c_array_, false));

        query_ = std::shared_ptr<tiledb::Query>(new Query(ctx_, *array_, TILEDB_READ));

        include_coords_ = include_coords;
    }


    void submit() {
        map<string, AttrInfo> buffers;
    }

    /* Data conversion routines */

    template <typename T>
    void convert(py::object o, vector<char>& buf)
    {
        buf.resize(sizeof(T));
        ((T*)static_cast<void*>(buf.data()))[0] = o.cast<T>();
    }

    template<>
    void convert<std::string>(py::object o, vector<char>& buf)
    {
        auto str = o.cast<std::string>();
        buf.resize(str.size());
        memcpy(buf.data(), str.c_str(), str.size());
    }

    void convert(py::object o, vector<char>& buf){
        if (py::isinstance<py::str>(o)) {
            convert<std::string>(o, buf);
        } else if (py::isinstance<py::float_>(o)) {
            convert<double>(o, buf);
        } else if (py::isinstance<py::int_>(o)) {

        }
    }

    void convert_type(py::object o, vector<char>& buf, tiledb_datatype_t type) {
        if (tiledb::impl::tiledb_string_type(type)) {
            convert<string>(o, buf);
        } else if (tiledb::impl::tiledb_datetime_type(type)) {
            convert<int64_t>(o, buf);
        } else if (type == TILEDB_FLOAT32) {
            convert<float>(o, buf);
        } else if (type == TILEDB_FLOAT64) {
            convert<double>(o, buf);
        }
    }

    void add_dim_range(uint32_t dim_idx, py::tuple r) {
        if (py::len(r) == 0)
            return;
        else if (py::len(r) != 2)
            TPY_ERROR_LOC("Unexpected range len != 2");

        auto r0 = r[0];
        auto r1 = r[1];
        if (r0.get_type() != r1.get_type())
            TPY_ERROR_LOC("Mismatched type");

        auto domain = array_->schema().domain();
        auto dim = domain.dimension(dim_idx);
        std::cout << dim << std::endl;

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
            case TILEDB_INT64:
            case TILEDB_INT8:
            case TILEDB_UINT8: {
                using T = uint8_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                }
            case TILEDB_INT16: {
                using T = int16_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                }
            case TILEDB_UINT16: {
                using T = uint16_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                }
            case TILEDB_UINT32: {
                using T = uint32_t;
                query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
                }
            case TILEDB_UINT64:
                query_->add_range(dim_idx, r0.cast<uint64_t>(), r1.cast<uint64_t>());
            case TILEDB_FLOAT32:
                query_->add_range(dim_idx, r0.cast<float>(), r1.cast<float>());
            case TILEDB_FLOAT64:
                query_->add_range(dim_idx, r0.cast<double>(), r1.cast<double>());
            case TILEDB_STRING_ASCII:
            case TILEDB_STRING_UTF8:
            case TILEDB_CHAR:
                if (!py::isinstance<py::str>(r0))
                    TPY_ERROR_LOC("internal error: expected string type for var-length dim!");
                query_->add_range(dim_idx, r0.cast<string>(), r1.cast<string>());
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
};


PYBIND11_MODULE(readquery, m) {
    py::class_<ReadQuery>(m, "ReadQuery")
        .def(py::init<py::object, py::object, py::tuple, bool>())
        .def("set_ranges", &ReadQuery::set_ranges)
        .def("submit", &ReadQuery::submit)
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
