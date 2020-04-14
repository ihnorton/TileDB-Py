#include <vector>
#include <string>
#include <iostream>

#include <pybind11/pybind11.h>

#include <tiledb/tiledb.h> // C

namespace py = pybind11;

namespace tiledbpy {

class ReadQuery {
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

    private:
        tiledb_ctx_t* ctx_;
        tiledb_array_t* array_;
        tiledb_query_t* query_;
        std::vector<std::string> attrs;
        bool include_coords_;

};

}; // namespace tiledbpy
