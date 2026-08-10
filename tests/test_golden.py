# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression against a stored golden, and a canary for the LFS round-trip.

The golden file is the only LFS object in the tree, so this is also how the sync
rehearsal finds out whether LFS survived the trip. A mirror that replays commits
but never pushes the LFS objects still *looks* correct — the pointer file is
ordinary versioned text — and the failure only shows up when something reads the
contents. That something is this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_cuda import vector_add

GOLDEN_PATH = Path(__file__).parent / "data" / "vector_add_golden.pt"

# What git-lfs leaves on disk when the object was never fetched.
LFS_POINTER_PREFIX = b"version https://git-lfs"


def test_golden_file_is_not_an_unfetched_lfs_pointer():
    head = GOLDEN_PATH.read_bytes()[: len(LFS_POINTER_PREFIX)]
    assert head != LFS_POINTER_PREFIX, (
        f"{GOLDEN_PATH.name} is an LFS pointer, not its contents. Either this "
        "clone skipped LFS (`git lfs pull`), or the publish replayed commits "
        "without pushing LFS objects to the destination — which is the failure "
        "this test exists to catch."
    )


def test_vector_add_matches_the_golden():
    if GOLDEN_PATH.read_bytes()[: len(LFS_POINTER_PREFIX)] == LFS_POINTER_PREFIX:
        pytest.skip("golden is an unfetched LFS pointer; see the canary test above")

    golden = torch.load(GOLDEN_PATH, weights_only=True)
    torch.testing.assert_close(vector_add(golden["a"], golden["b"]), golden["expected"])
