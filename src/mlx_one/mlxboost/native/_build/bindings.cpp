// MLX nanobind API adaptation. Copyright © 2023-2025 Apple Inc. MIT.
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include "axpby.h"
#include "gemv.h"
#include "build_identity.h"
namespace nb = nanobind;
using namespace nb::literals;
void bind_replay(nb::module_&);
NB_MODULE(_ext, m) {
  bind_replay(m);
  m.attr("mlx_version") = MLXBOOST_MLX_VERSION;
  m.attr("mlx_build_sha256") = MLXBOOST_MLX_SHA;
  m.attr("implementation_sha256") = MLXBOOST_SOURCE_SHA;
  m.attr("build_python") = MLXBOOST_BUILD_PYTHON;
  m.def("axpby", &mlxboost::axpby, "x"_a, "y"_a, "alpha"_a, "beta"_a,
        nb::kw_only(), "threadgroup_size"_a = 128, "stream"_a = nb::none());
  m.def("max_threads", &mlxboost::max_threads, "dtype"_a, nb::kw_only(), "stream"_a = nb::none());
  nb::class_<mlxboost::GemvTrace>(m, "GemvTrace")
      .def(nb::init<>())
      .def_prop_ro("dispatches", [](const mlxboost::GemvTrace& trace) {
        return trace.dispatches.load(std::memory_order_relaxed);
      });
  m.def("quantized_gemv", &mlxboost::quantized_gemv,
        "x"_a, "weight"_a, "scales"_a, "biases"_a, nb::kw_only(),
        "bits"_a, "group_size"_a, "threadgroup_size"_a = 128, "stream"_a = nb::none(),
        "trace"_a = nb::none());
  m.def("gemv_max_threads", &mlxboost::gemv_max_threads, "dtype"_a, "bits"_a,
        "group_size"_a, nb::kw_only(), "stream"_a = nb::none());
  m.def("layout", [](const mlxboost::mx::array& x) {
    nb::dict d;
    d["row_contiguous"] = static_cast<bool>(x.flags().row_contiguous);
    // MLX Strides is a SmallVector, not a registered nanobind std::vector.
    nb::list strides;
    for (auto stride : x.strides()) strides.append(nb::int_(stride));
    d["strides"] = strides;
    return d;
  });
}
