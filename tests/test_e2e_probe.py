# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Deliberately failing probe used by the end-to-end sync test."""

from __future__ import annotations


def test_e2e_probe_fails():
    assert 1 == 2, "deliberate e2e failure"
