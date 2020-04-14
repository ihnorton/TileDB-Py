#include <vector>
#include <memory>
#include <string>
#include <iostream>


#include <tiledb/tiledb> // C++
#include <tiledb/tiledb.h> // C

#include "readquery.h"

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

using namespace tiledb;
using namespace std;
namespace py = pybind11;

namespace tiledbpy {

class TileDBPyError : std::runtime_error {
public:
    explicit TileDBPyError(const char * m) : std::runtime_error(m) , message{m} {}

    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

void ReadQuery::test(py::tuple ex) {
    for (auto o : ex) {
        py::print(o);
    }
    throw TileDBPyError("foobar!");
}

ReadQuery::ReadQuery(
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

struct RQBuffer {
    string name;
    vector<char> data;
    vector<char> offsets;
}

void ReadQuery::submit() {
    map<string, RQBuffer> buffers;


}


void ReadQuery::set_ranges(py::iterable ranges) {
    // ranges are specified as one iterable per dimension

    for (auto dim_range : ranges) {
        py::print(dim_range);
    }

}


PYBIND11_MODULE(readquery, m) {
    py::class_<ReadQuery>(m, "ReadQuery")
        .def(py::init<py::object, py::object, py::tuple, bool>())
        .def("set_ranges", &ReadQuery::set_ranges)
        .def("submit", &ReadQuery::submit)
        .def("test", &ReadQuery::test);

    /*
       we need to make sure C++ TileDBError is translated to a correctly-typed py error.
       using py::exception(..., "TileDBError") creates a new exception
       in the *readquery* module.
    */
    static auto tiledb_py_error = (py::object) py::module::import("tiledb").attr("TileDBError");

    py::register_exception_translator([](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            } catch (const TileDBError &e) {
                // TODO: set C++ line number if possible
                PyErr_SetString(tiledb_py_error.ptr(), e.what());
            }
            catch (std::exception &e) {
                std::cout << "got some other error" << e.what();
            }
        });
}

}; // namespace tiledbpy
