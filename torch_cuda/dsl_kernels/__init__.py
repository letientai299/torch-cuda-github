# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Kernel sources.

`cute/` is ahead-of-time compiled and is the IP the publish pipeline is meant to
withhold; `triton/` is JIT-compiled at call time and therefore has to ship.
Nothing here is imported at package import time — the ops import lazily so the
public package stays importable when `cute/` has been filtered out.
"""
