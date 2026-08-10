/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cubin_runtime.h"

// Generated into the build tree by cpp/tools/embed_cubins.py; see kernels/CMakeLists.txt.
#include "embedded_cubins.h"

#include <cuda.h>

#include <map>
#include <mutex>
#include <stdexcept>

namespace torch_cuda
{
namespace
{

void check(CUresult result, char const* what)
{
  if (result == CUDA_SUCCESS)
  {
    return;
  }
  char const* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(what) + ": " + (message ? message : "unknown CUDA error"));
}

int current_sm_arch()
{
  CUdevice device{};
  check(cuCtxGetDevice(&device), "cuCtxGetDevice");
  int major = 0;
  int minor = 0;
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "compute capability major");
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "compute capability minor");
  return major * 10 + minor;
}

} // namespace

void* CubinRuntime::function(std::string const& kernel_name)
{
  // Guards the cache below; nanobind releases the GIL around launches, so two
  // Python threads can reach this concurrently on first use.
  static std::mutex mutex;
  static std::map<std::string, CUfunction> cache;
  std::lock_guard<std::mutex> const lock(mutex);

  std::string const key = kernel_name + "@" + std::to_string(current_sm_arch());
  if (auto const it = cache.find(key); it != cache.end())
  {
    return it->second;
  }

  CubinImage const* image = find_embedded_cubin(kernel_name, current_sm_arch());
  if (image == nullptr)
  {
    throw std::runtime_error("no embedded CUBIN for " + key);
  }

  CUmodule module{};
  check(cuModuleLoadData(&module, image->data), "cuModuleLoadData");
  CUfunction fn{};
  check(cuModuleGetFunction(&fn, module, image->entry), "cuModuleGetFunction");

  cache.emplace(key, fn);
  return fn;
}

bool CubinRuntime::has(std::string const& kernel_name, int sm_arch) noexcept
{
  return find_embedded_cubin(kernel_name, sm_arch) != nullptr;
}

} // namespace torch_cuda
