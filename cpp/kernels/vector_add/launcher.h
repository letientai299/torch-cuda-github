/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace torch_cuda
{

/// Launches the embedded elementwise-add CUBIN on `stream`.
///
/// Pointers are device addresses of contiguous float32 buffers of `n_elements`;
/// the caller (bindings.cpp) owns validation.
void launch_vector_add(void const* a, void const* b, void* out, std::size_t n_elements, std::uintptr_t stream);

} // namespace torch_cuda
