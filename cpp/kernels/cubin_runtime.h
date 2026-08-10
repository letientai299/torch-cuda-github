/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace torch_cuda
{

/// One ahead-of-time compiled kernel image, tagged with the SM it was built for.
struct CubinImage
{
  int sm_arch;
  /// The symbol `cuModuleGetFunction` needs — the DSL's mangled name, not the
  /// logical kernel name used to look this image up. Read out of the CUBIN by
  /// cpp/tools/embed_cubins.py.
  char const* entry;
  void const* data;
  std::size_t size;
};

/// Loads an embedded CUBIN for the current device and hands back a launchable
/// function. Modules are cached per (device, kernel) — cuModuleLoadData is
/// expensive enough that reloading it per launch would dominate a small kernel.
class CubinRuntime
{
  public:
  /// Returns the CUfunction (as an opaque handle) for `kernel_name`, selecting
  /// the image whose sm_arch matches the current device.
  ///
  /// Throws std::runtime_error when no image matches, which is the case a
  /// consumer hits on an unsupported GPU; the Python layer turns it into a
  /// backend fallback rather than a crash.
  static void* function(std::string const& kernel_name);

  /// Whether an image for `kernel_name` exists for `sm_arch`.
  ///
  /// Importing the extension proves only that it was built, not that it carries
  /// SASS this GPU can run: a build with no images, or one that shipped other
  /// architectures, links and imports exactly the same. Callers use this to
  /// choose a backend before committing to a launch that would otherwise throw.
  ///
  /// `sm_arch` is a parameter rather than something queried here because this
  /// runs during backend selection, before anything has made a CUDA context
  /// current — and `cuCtxGetDevice` fails without one. Python already knows the
  /// capability via torch, so it passes it in. A pure table lookup also cannot
  /// fail, which is what a probe should be.
  static bool has(std::string const& kernel_name, int sm_arch) noexcept;
};

} // namespace torch_cuda
