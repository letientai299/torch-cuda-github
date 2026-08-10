/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vector_add/launcher.h"

#include "cubin_runtime.h"

#include <cuda.h>

#include <stdexcept>

namespace torch_cuda
{
namespace
{

// Must match THREADS_PER_BLOCK in torch_cuda/dsl_kernels/cute/sm80_vector_add.py.
// The CUBIN bakes the block shape in, so the two constants are one value split
// across the build boundary; prepare_cubins.py re-emits it into the generated
// header so a mismatch fails the build rather than the run.
constexpr unsigned int kThreadsPerBlock = 256;

} // namespace

void launch_vector_add(void const* a, void const* b, void* out, std::size_t n_elements, std::uintptr_t stream)
{
  if (n_elements == 0)
  {
    return;
  }

  auto* fn = static_cast<CUfunction>(CubinRuntime::function("vector_add_kernel"));

  auto n = static_cast<int>(n_elements);
  void* args[] = {&a, &b, &out, &n};
  unsigned int blocks = static_cast<unsigned int>((n_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);

  CUresult const result
    = cuLaunchKernel(fn, blocks, 1, 1, kThreadsPerBlock, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr);
  if (result != CUDA_SUCCESS)
  {
    char const* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string("cuLaunchKernel: ") + (message ? message : "unknown"));
  }
}

} // namespace torch_cuda
