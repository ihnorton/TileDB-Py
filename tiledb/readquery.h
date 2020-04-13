#include <vector>
#include <string>
#include <iostream>

#include <pybind11/pybind11.h>

#include <tiledb/tiledb.h> // C

namespace py = pybind11;

namespace tiledbpy {

struct RQ;

class ReadQuery {
    public:
        ReadQuery();
        ReadQuery(
            py::object ctx_cap,
            py::object array_cap,
            py::tuple attrs,
            bool include_coords
        );

        void test(py::tuple x);

        void set_ranges(py::tuple ranges);

    private:
        std::unique_ptr<RQ> rq_;

};

}; // namespace tiledbpy
