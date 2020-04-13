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

struct RQ {
    RQ() = default;

    tiledb_ctx_t* ctx;
    tiledb_array_t* array;
    tiledb_query_t* query;
};

ReadQuery::ReadQuery() {
    rq_ = unique_ptr<RQ>();
};

ReadQuery::ReadQuery(
    py::capsule ctx_cap,
    py::object array_cap,
    py::tuple attrs,
    bool include_coords) {

}

void ReadQuery::test(py::tuple ex) {
    for (auto o : ex) {
        //std::cout << o << std::endl;
        py::print(o);
    }
}

void ReadQuery::set_ranges(py::tuple ranges) {

}

PYBIND11_MODULE(readquery, m) {
    py::class_<ReadQuery>(m, "ReadQuery")
        .def(py::init())
        .def(py::init<py::object, py::object, py::tuple, bool>())
        .def("test", &ReadQuery::test);
}

}; // namespace tiledbpy

/*
PYBIND11_MODULE(readquery, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test", &tiledbpy::test, "a function that prints tuple");
}
*/

