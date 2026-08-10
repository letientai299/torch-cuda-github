# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Install location of the compiled nanobind extension.

The directory is otherwise gitignored: `setup.py` drops `_kernels*.so` here at
build time. Keeping the package marker committed means an editable install
without the extension still imports.
"""
