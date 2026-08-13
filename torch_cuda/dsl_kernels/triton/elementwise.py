# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Elementwise Triton kernels and their launchers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One warp-friendly tile per program. Elementwise work is bandwidth-bound, so a
# large block keeps the memory pipeline full without needing autotuning here.
BLOCK_SIZE = 1024


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.jit
def _scale_add_kernel(x_ptr, y_ptr, out_ptr, alpha, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * alpha + y, mask=mask)


def triton_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    _add_kernel[_grid(a.numel())](a, b, out, a.numel(), BLOCK=BLOCK_SIZE)
    return out


def triton_fused_scale_add(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(x)
    _scale_add_kernel[_grid(x.numel())](x, y, out, alpha, x.numel(), BLOCK=BLOCK_SIZE)
    return out


def _grid(n_elements: int) -> tuple[int, ...]:
    return (triton.cdiv(n_elements, BLOCK_SIZE),)


# a comment added inside the published tree instead
