# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Op-level behaviour, checked against the eager reference.

Every backend has to agree with `a + b`; on a GPU runner the same cases run
again through whichever accelerated backend `select_backend` picks, so a CUBIN
that drifts from the reference fails here rather than in a consumer's model.
"""

from __future__ import annotations

import pytest
import torch

from torch_cuda import fused_scale_add, vector_add
from torch_cuda.runtime import Backend, select_backend

SHAPES = [(1,), (1024,), (33, 65), (4, 8, 16)]


@pytest.mark.parametrize("shape", SHAPES)
def test_vector_add_matches_reference(shape):
    a = torch.randn(shape)
    b = torch.randn(shape)
    torch.testing.assert_close(vector_add(a, b), a + b)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("alpha", [0.0, 1.0, -2.5])
def test_fused_scale_add_matches_reference(shape, alpha):
    x = torch.randn(shape)
    y = torch.randn(shape)
    torch.testing.assert_close(fused_scale_add(x, y, alpha), alpha * x + y)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        (torch.zeros(4), torch.zeros(5), "shape mismatch"),
        (torch.zeros(4), torch.zeros(4, dtype=torch.float64), "dtype mismatch"),
        (torch.zeros(4, 4).t(), torch.zeros(4, 4), "contiguous"),
    ],
)
def test_vector_add_rejects_bad_operands(a, b, message):
    with pytest.raises(ValueError, match=message):
        vector_add(a, b)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_accelerated_backend_matches_eager():
    a = torch.randn(4096, device="cuda")
    b = torch.randn(4096, device="cuda")
    assert select_backend() is not Backend.EAGER
    torch.testing.assert_close(vector_add(a, b), vector_add(a, b, backend=Backend.EAGER))
