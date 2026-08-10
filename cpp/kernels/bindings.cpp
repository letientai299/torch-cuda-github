/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cubin_runtime.h"
#include "vector_add/launcher.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// Registers the str <-> std::string caster. Without it nanobind has no
// converter for has_cubin's argument and rejects a plain Python str at call
// time, not at build time.
#include <nanobind/stl/string.h>

#include <stdexcept>

namespace nb = nanobind;

namespace
{

using DeviceArray = nb::ndarray<float, nb::device::cuda, nb::c_contig>;

void vector_add(DeviceArray const& a, DeviceArray const& b, DeviceArray& out, std::uintptr_t stream)
{
  if (a.size() != b.size() || a.size() != out.size())
  {
    throw std::invalid_argument("vector_add: operands and output must have equal element counts");
  }
  torch_cuda::launch_vector_add(a.data(), b.data(), out.data(), a.size(), stream);
}

} // namespace

NB_MODULE(_kernels, m)
{
  m.doc() = "Pre-compiled CUDA kernels. The SASS is embedded; no sources ship with the wheel.";
  // stream defaults to the legacy default stream; torch_cuda.ops passes the
  // current torch stream so the launch orders against the caller's work.
  m.def("vector_add", &vector_add, nb::arg("a"), nb::arg("b"), nb::arg("out"), nb::arg("stream") = 0);
  // Lets the Python backend chain ask whether this build can actually serve the
  // current device, instead of inferring it from the import succeeding.
  m.def("has_cubin", &torch_cuda::CubinRuntime::has, nb::arg("kernel_name"), nb::arg("sm_arch"));
}
