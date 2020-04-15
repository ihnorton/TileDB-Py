#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#undef NPY_NO_DEPRECATED_API

#define PY_ERROR_LOC(m) TileDBPyError(m, __FILE__, __LINE__);

//#include <tiledb/tiledb> // C++
//using namespace tiledb;

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

namespace py = pybind11;

namespace tiledbpy {

using namespace std;

class TileDBPyError : std::runtime_error {
public:
    explicit TileDBPyError(const char * m) : std::runtime_error(m) , message{m} {}
    explicit TileDBPyError(std::string m) : std::runtime_error(m.c_str()), message{m.c_str()} {}
    explicit TileDBPyError(std::string m, const char *file, int line))
        : std::runtime_error(m.c_str())
        , message{m.c_str()} {}
public:
    virtual const char * what() const noexcept override {return message.c_str();}

private:
    std::string message = "";
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
        tiledb_ctx_t* ctx_;
        tiledb_array_t* array_;
        tiledb_query_t* query_;
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
        throw TileDBPyError("foobar!");
    }

    ReadQuery(
        py::object ctx,
        py::object array,
        py::tuple attrs,
        bool include_coords) {

        ctx_ = (py::capsule)ctx.attr("__capsule__")();
        array_ = (py::capsule)array.attr("__capsule__")();

        query_ = nullptr;
        auto rc = tiledb_query_alloc(ctx_, array_, TILEDB_READ, &query_);
        if (rc != TILEDB_OK)
            throw TileDBPyError("Failed to allocate query");

        include_coords_ = include_coords;
    }


    void submit() {
        map<string, AttrInfo> buffers;
    }

    void add_dim_range(uint32_t dim_idx, py::tuple r) {
        if (py::len(r) == 0)
            return;
        else if (py::len(r) != 2)
            throw PY_ERROR_LOC("Unexpected range len != 2");

        auto r0 = r[0];
        auto r1 = r[1];
        if (r0.get_type() != r1.get_type())
            throw TileDBPyError("Mismatched type");

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
        .def("test", &ReadQuery::test);

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
            }
            catch (std::exception &e) {
                std::cout << "got some other error" << e.what();
            }
        });
}

}; // namespace tiledbpy
