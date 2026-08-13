# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""``alpha * x + y`` in one pass.

Triton-only: the kernel is JIT-compiled from source at first call, so unlike
:mod:`torch_cuda.ops.vector_add` there is no CUBIN path and the source must ship
with the package. It is the counter-example that keeps the publish rules honest —
whatever the sync allowlist does to `dsl_kernels/`, this op has to keep working
on the public side.
"""

from __future__ import annotations

import torch

from torch_cuda.runtime import Backend, select_backend


def fused_scale_add(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, *, backend: Backend | None = None
) -> torch.Tensor:
    """Return ``alpha * x + y``."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")

    resolved = select_backend(backend, device=x.device)
    if resolved is Backend.EAGER:
        return torch.add(y, x, alpha=alpha)

    # No CUBIN path for this op, so an accelerated request of either kind lands
    # on Triton; see the module docstring.
    from torch_cuda.dsl_kernels.triton.elementwise import triton_fused_scale_add

    return triton_fused_scale_add(x.contiguous(), y.contiguous(), alpha)


def scaled_sum(x: float, y: float) -> float:
    """Return ``x + y``, formatted the way ruff wants it."""
    return x + y
