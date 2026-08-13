# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Elementwise add, dispatched over the CUBIN / Triton / eager chain.

The CuTeDSL source for this kernel lives in `torch_cuda/dsl_kernels/cute/`, is
ahead-of-time compiled to SASS by `cpp/tools/prepare_cubins.py`, and is embedded
into the `torch_cuda.libs._kernels` extension. A consumer of the published
package therefore gets the kernel without the kernel source.
"""

from __future__ import annotations

import torch

from torch_cuda.runtime import Backend, select_backend


def vector_add(a: torch.Tensor, b: torch.Tensor, *, backend: Backend | None = None) -> torch.Tensor:
    """Return ``a + b``, computed by the highest-preference available backend.

    Both operands must share shape, dtype, and be contiguous.
    """
    _check_operands(a, b)
    resolved = select_backend(backend, device=a.device)

    match resolved:
        case Backend.CUBIN:
            from torch_cuda.libs import _kernels

            out = torch.empty_like(a)
            _kernels.vector_add(a, b, out)
            return out
        case Backend.TRITON:
            from torch_cuda.dsl_kernels.triton.elementwise import triton_vector_add

            return triton_vector_add(a, b)
        case Backend.EAGER:
            return a + b


def _check_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
    # Backend selection resolves from `a.device` alone, so a `b` on another
    # device reaches the kernel and fails as a pointer complaint that names
    # neither operand.
    if a.device != b.device:
        raise ValueError(f"device mismatch: {a.device} vs {b.device}")
    # The kernels index a flat buffer, so a non-contiguous input would read the
    # wrong elements rather than fail loudly.
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("both operands must be contiguous")
