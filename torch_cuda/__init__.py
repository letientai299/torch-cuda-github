# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Public frontend of the sandbox package.

This mirrors the layering of the production repo: the PyTorch-facing API lives
in :mod:`torch_cuda.ops`, kernel sources live in :mod:`torch_cuda.dsl_kernels`,
and the compiled-artifact loader lives in :mod:`torch_cuda.runtime`. Only the
first and last are meant to reach the public mirror.
"""

from torch_cuda.ops import fused_scale_add, vector_add

__all__ = ["fused_scale_add", "vector_add"]
