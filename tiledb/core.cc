#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#undef NPY_NO_DEPRECATED_API

#define TPY_ERROR_LOC(m)                                                       \
  throw TileDBPyError(std::string(m) + " (" + __FILE__ + ":" +                 \
                      std::to_string(__LINE__) + ")");

#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
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

const uint64_t DEFAULT_INIT_BUFFER_BYTES = 1310720 * 8;
const uint64_t DEFAULT_EXP_ALLOC_MAX_BYTES = 4 * pow(2, 30);

class TileDBPyError : std::runtime_error {
public:
  explicit TileDBPyError(const char *m) : std::runtime_error(m) {}
  explicit TileDBPyError(std::string m) : std::runtime_error(m.c_str()) {}

public:
  virtual const char *what() const noexcept override {
    return std::runtime_error::what();
  }
};

static void pyprint(py::object o) {
    py::print(o);
}

py::dtype tiledb_dtype(tiledb_datatype_t type);

struct BufferInfo {

  BufferInfo(std::string name, tiledb_datatype_t type, size_t elem_bytes,
             size_t offsets_num, bool isvar = false)
      : name(name), type(type), isvar(isvar) {
    py::dtype dtype = tiledb_dtype(type);
    data = py::array(dtype, elem_bytes / dtype.itemsize());
    offsets = py::array_t<uint64_t>(offsets_num);
  }

  string name;
  tiledb_datatype_t type;
  py::array data;
  py::array_t<uint64_t> offsets;
  uint64_t data_read = 0;
  uint64_t offsets_read = 0;
  bool isvar;
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
  Context ctx_;
  shared_ptr<tiledb::Array> array_;
  shared_ptr<tiledb::Query> query_;
  std::vector<std::string> attrs_;
  map<string, BufferInfo> buffers_;
  bool include_coords_;
  uint64_t init_buffer_bytes_ = DEFAULT_INIT_BUFFER_BYTES;
  uint64_t exp_alloc_max_bytes_ = DEFAULT_EXP_ALLOC_MAX_BYTES;

public:
  tiledb_ctx_t *c_ctx_;
  tiledb_array_t *c_array_;

public:
  PyQuery() = delete;

  PyQuery(py::object ctx, py::object array, py::tuple attrs,
          bool include_coords) {

    tiledb_ctx_t *c_ctx_ = (py::capsule)ctx.attr("__capsule__")();
    if (c_ctx_ == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!")
    ctx_ = Context(c_ctx_, false);

    tiledb_array_t *c_array_ = (py::capsule)array.attr("__capsule__")();

    /* TBD whether we use the C++ API ... */
    // we never own this pointer, pass own=false
    array_ = std::shared_ptr<tiledb::Array>(new Array(ctx_, c_array_, false),
                                            [](Array *p) {} /* no deleter*/);

    query_ = std::shared_ptr<tiledb::Query>(
        new Query(ctx_, *array_, TILEDB_READ)); //,
    //                     [](Query* p){} /* no deleter*/);

    include_coords_ = include_coords;

    for (auto a : attrs) {
      attrs_.push_back(a.cast<string>());
    }

    // get config parameters
    try {
      init_buffer_bytes_ =
          std::atoi(ctx_.config().get("py.init_buffer_bytes").c_str());
    } catch (TileDBError &e) { /* pass, key not found */ }
    try {
      exp_alloc_max_bytes_ =
          std::atoi(ctx_.config().get("py.exp_alloc_max_bytes").c_str());
    } catch (TileDBError &e) {
      /* pass, key not found */
    }
  }

  void add_dim_range(uint32_t dim_idx, py::tuple r) {
    if (py::len(r) == 0)
      return;
    else if (py::len(r) != 2)
      TPY_ERROR_LOC("Unexpected range len != 2");

    auto r0 = r[0];
    auto r1 = r[1];
    // no type-check here, because we might allow cast-conversion
    // if (r0.get_type() != r1.get_type())
    //    TPY_ERROR_LOC("Mismatched type");

    auto domain = array_->schema().domain();
    auto dim = domain.dimension(dim_idx);

    auto tiledb_type = dim.type();

    try {
      switch (tiledb_type) {
      case TILEDB_INT32: {
        using T = int32_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_INT64: {
        using T = int64_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_INT8: {
        using T = uint8_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT8: {
        using T = uint8_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_INT16: {
        using T = int16_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT16: {
        using T = uint16_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT32: {
        using T = uint32_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT64: {
        using T = uint64_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_FLOAT32: {
        using T = float;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_FLOAT64: {
        using T = double;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_STRING_ASCII:
      case TILEDB_STRING_UTF8:
      case TILEDB_CHAR: {
        if (!py::isinstance<py::str>(r0))
          TPY_ERROR_LOC(
              "internal error: expected string type for var-length dim!");
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
      std::string msg = "Failed to cast dim range '" + (string)py::repr(r) +
                        "' to dim type " +
                        tiledb::impl::type_to_str(tiledb_type);
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

  bool is_var(std::string name) {
    auto domain = array_->schema().domain();
    if (domain.has_dimension(name)) {
      auto dim = domain.dimension(name);
      return dim.cell_val_num() == TILEDB_VAR_NUM;
    } else if (array_->schema().has_attribute(name)) {
      auto attr = array_->schema().attribute(name);
      return attr.cell_val_num() == TILEDB_VAR_NUM;
    } else {
      TPY_ERROR_LOC("Unknown buffer type for is_var check (expected attribute "
                    "or dimension)")
    }
  }

  void set_buffer(py::str name, py::object data) {
    // set input data for an attribute or dimension buffer
    if (array_->schema().domain().has_dimension(name))
      set_buffer(name, data);
    else if (array_->schema().has_attribute(name))
      set_buffer(name, data);
    else
      TPY_ERROR_LOC("Unknown attr or dim '" + (string)name + "'")
  }

  bool is_sparse() { return array_->schema().array_type() == TILEDB_SPARSE; }

  void alloc_buffer(std::string name, tiledb_datatype_t type) {
    auto schema = array_->schema();
    uint64_t buf_bytes = 0;
    uint64_t offsets_num = 0;
    bool var = is_var(name);

    if (var) {
      auto size_pair = query_->est_result_size_var(name);
      buf_bytes = size_pair.second;
      offsets_num = size_pair.first;
    } else {
      buf_bytes = query_->est_result_size(name);
    }

    if ((var || is_sparse()) && buf_bytes < init_buffer_bytes_) {
      buf_bytes = init_buffer_bytes_;
      offsets_num = init_buffer_bytes_ / sizeof(uint64_t);
    }

    buffers_.insert({name, BufferInfo(name, type, buf_bytes, offsets_num, var)});
  }

  void set_buffers() {
    for (auto bp : buffers_) {
      auto name = bp.first;
      auto b = bp.second;
      if (b.isvar) {
        query_->set_buffer(b.name, (uint64_t *)b.offsets.data(),
                           b.offsets.size(), (void *)b.data.data(),
                           b.data.size());
      } else {
        query_->set_buffer(b.name, (void *)b.data.data(), b.data.size());
      }
    }
  }

  void update_read_elem_num() {
    for (const auto &read_info : query_->result_buffer_elements()) {
      auto name = read_info.first;
      uint64_t offset_elem_num, data_elem_num;
      std::tie(offset_elem_num, data_elem_num) = read_info.second;
      BufferInfo &buf = buffers_.at(name);
      buf.data_read += data_elem_num;
      buf.offsets_read += offset_elem_num;
    }
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

    set_buffers();

    size_t max_retries = 0, retries = 0;
    try {
      max_retries =
          std::atoi(ctx_.config().get("py.max_incomplete_retries").c_str());
    } catch (tiledb::TileDBError &e) {
      max_retries = 100;
    }

    {
      py::gil_scoped_release release;
      query_->submit();
    }

    // TODO: would be nice to have a callback here to customize the reallocation
    // strategy
    while (query_->query_status() == Query::Status::INCOMPLETE) {
      if (++retries > max_retries)
        TPY_ERROR_LOC(
            "Exceeded maximum retries ('py.max_incomplete_retries': " +
            std::to_string(max_retries) + "')");

      if (is_sparse()) {
        // TODO do we also need to realloc for var-length queries?
        for (auto bp : buffers_) {
          auto buf = bp.second;
          buf.data.resize({buf.data.size() * 2});

          if (buf.isvar)
            buf.offsets.resize({buf.offsets.size() * 2});
        }
      }
      {
        py::gil_scoped_release release;
        query_->submit();
      }

      update_read_elem_num();
    }

    update_read_elem_num();

    for (auto bp : buffers_) {
      auto name = bp.first;
      auto &buf = bp.second;
      buf.data.resize({buf.data_read});
      buf.offsets.resize({buf.offsets_read});
    }
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

  py::dict results() {
    py::dict results;
    for (auto &bp : buffers_) {
      results[py::str(bp.first)] =
          py::make_tuple(bp.second.data, bp.second.offsets);
    }
    return results;
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
      .def("results", &PyQuery::results)
      .def("test_array", &PyQuery::test_array)
      .def("test_err",
           [](py::object self, std::string s) { throw TileDBPyError(s); });

  /*
     We need to make sure C++ TileDBError is translated to a correctly-typed py
     error. Note that using py::exception(..., "TileDBError") creates a new
     exception in the *readquery* module, so we must import to reference.
  */
  static auto tiledb_py_error =
      (py::object)py::module::import("tiledb").attr("TileDBError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const TileDBPyError &e) {
      PyErr_SetString(tiledb_py_error.ptr(), e.what());
    } catch (const tiledb::TileDBError &e) {
      PyErr_SetString(tiledb_py_error.ptr(), e.what());
    } catch (std::exception &e) {
      std::cout << "got some other error" << e.what();
    }
  });
}

}; // namespace tiledbpy
