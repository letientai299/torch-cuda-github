# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Backend resolution — the part of the package that has to behave identically
on a developer box, on internal CI with GPUs, and on public CI without one.

Nothing here is skipped on a GPU machine. An earlier version skipped the whole
module whenever CUDA was available, which hid the case that actually broke: CPU
tensors on a machine that happens to have a GPU.
"""

from __future__ import annotations

import pytest
import torch

from torch_cuda.runtime import Backend, select_backend

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA device")


def test_cpu_operands_resolve_to_eager_even_with_a_gpu_present():
    """The regression: availability is about the operands, not the machine."""
    assert select_backend(device=torch.device("cpu")) is Backend.EAGER


@pytest.mark.parametrize("unavailable", [Backend.CUBIN, Backend.TRITON])
def test_accelerated_backend_requested_for_cpu_operands_raises(unavailable):
    with pytest.raises(RuntimeError, match="not available for a cpu tensor"):
        select_backend(unavailable, device=torch.device("cpu"))


def test_env_override_is_validated(monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_BACKEND", "nonsense")
    with pytest.raises(RuntimeError, match="not one of"):
        select_backend()


def test_env_override_selects_eager(monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_BACKEND", "eager")
    assert select_backend() is Backend.EAGER


def test_env_override_cannot_force_a_device_backend_onto_cpu_operands(monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_BACKEND", "triton")
    with pytest.raises(RuntimeError, match="not available for a cpu tensor"):
        select_backend(device=torch.device("cpu"))


@pytest.mark.skipif(CUDA_AVAILABLE, reason="asserts the no-GPU fallback chain")
def test_without_a_gpu_everything_falls_back_to_eager():
    assert select_backend() is Backend.EAGER
    for unavailable in (Backend.CUBIN, Backend.TRITON):
        with pytest.raises(RuntimeError, match="not available"):
            select_backend(unavailable)


@requires_cuda
@pytest.mark.gpu
def test_cuda_operands_resolve_to_an_accelerated_backend():
    resolved = select_backend(device=torch.device("cuda"))
    # CUBIN when the extension is built, Triton otherwise — never eager.
    assert resolved in (Backend.CUBIN, Backend.TRITON)
