# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Which implementation actually runs an op, and why.

The production repo ships pre-compiled CUBINs so consumers never need the kernel
sources or a CUDA toolkit. That makes backend choice a real, user-visible
fallback chain rather than an internal detail, so it is modelled as one status
value instead of a set of `has_cubin` / `use_triton` booleans.
"""

from __future__ import annotations

import functools
import os
from enum import StrEnum

import torch

_BACKEND_ENV = "TORCH_CUDA_BACKEND"


class Backend(StrEnum):
    """Implementation used to execute an op, in descending order of preference."""

    #: Pre-compiled SASS loaded from the nanobind extension. Ships without sources.
    CUBIN = "cubin"
    #: Triton kernel JIT-compiled on the host. Needs a GPU, not the kernel sources.
    TRITON = "triton"
    #: Plain PyTorch. Always available, including on CPU-only machines.
    EAGER = "eager"


#: Kernel the CUBIN backend needs; its presence stands for "this build is usable".
_PROBE_KERNEL = "vector_add_kernel"


@functools.cache
def _cubin_available_for(sm_arch: int) -> bool:
    """Cached per architecture — the answer differs across GPUs in one process."""
    try:
        from torch_cuda.libs import _kernels
    except ImportError:
        return False
    return _kernels.has_cubin(_PROBE_KERNEL, sm_arch)


def cubin_extension_available(device: torch.device | None = None) -> bool:
    """True when the extension is importable *and* carries SASS for this device.

    Importability alone is not enough. An extension built with no images, or
    built for architectures this GPU is not, imports exactly like a working one
    — and then throws on every launch. Treating that as "available" turns the
    fallback chain into a hard error, which is the opposite of its purpose.

    The architecture comes from torch rather than from the driver inside the
    extension: this runs during backend selection, before anything has made a
    CUDA context current, and the driver's device query needs one.
    """
    if not torch.cuda.is_available():
        return False
    index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return _cubin_available_for(major * 10 + minor)


def select_backend(requested: Backend | None = None, *, device: torch.device | None = None) -> Backend:
    """Resolve the backend for one call.

    `device` is where the operands actually live, and it is the deciding input:
    a machine having a GPU says nothing about whether *these* tensors are on it.
    Resolving from `torch.cuda.is_available()` alone routes CPU tensors into a
    device kernel, which fails inside the launcher with a message about pointers
    rather than about devices. Omitting `device` keeps the process-level answer,
    which is all a caller like `select_backend()` in a status line can know.

    An explicit `requested` value wins, then `TORCH_CUDA_BACKEND`, then the
    highest-preference backend whose prerequisites are met. An explicit request
    that cannot be satisfied raises rather than silently degrading — silent
    degradation is how a "kernel regression" turns out to be a packaging bug.
    """
    override = requested or _requested_from_env()
    if override is not None:
        if not _is_available(override, device):
            raise RuntimeError(
                f"backend {override.value!r} was requested but is not available"
                + (f" for a {device.type} tensor" if device is not None else " on this machine")
            )
        return override

    for backend in (Backend.CUBIN, Backend.TRITON):
        if _is_available(backend, device):
            return backend
    return Backend.EAGER


def _requested_from_env() -> Backend | None:
    raw = os.environ.get(_BACKEND_ENV, "").strip().lower()
    if not raw:
        return None
    try:
        return Backend(raw)
    except ValueError:
        valid = ", ".join(b.value for b in Backend)
        raise RuntimeError(f"{_BACKEND_ENV}={raw!r} is not one of: {valid}") from None


def _is_available(backend: Backend, device: torch.device | None = None) -> bool:
    # Operands off the GPU can only be served by eager, whatever the machine has
    # installed. Checked before the per-backend prerequisites because it
    # overrides all of them.
    if device is not None and device.type != "cuda":
        return backend is Backend.EAGER

    match backend:
        case Backend.CUBIN:
            return torch.cuda.is_available() and cubin_extension_available(device)
        case Backend.TRITON:
            return torch.cuda.is_available()
        case Backend.EAGER:
            return True
