# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Smallest end-to-end use of the public API: ``python examples/vector_add.py``."""

import torch

from torch_cuda import fused_scale_add, vector_add
from torch_cuda.runtime import select_backend


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} backend={select_backend().value}")

    a = torch.arange(8, dtype=torch.float32, device=device)
    b = torch.ones(8, device=device)

    print("vector_add     ", vector_add(a, b).tolist())
    print("fused_scale_add", fused_scale_add(a, b, alpha=2.0).tolist())


if __name__ == "__main__":
    main()

# outbound coverage: touched so every published path has a real last-commit message
