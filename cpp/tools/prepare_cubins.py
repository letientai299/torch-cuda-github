# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Ahead-of-time compile the CuTeDSL kernels into committed `.cubin` artifacts.

This is the seam that lets the kernel sources stay private: it runs inside the
internal pipeline, reads `torch_cuda/dsl_kernels/cute/`, and writes one `.cubin`
per matrix cell. Those files are committed (via LFS) and published, so the public
tree gets working kernels with no kernel sources and no nvcc.

Embedding them into the extension is a separate step — see `embed_cubins.py`,
which CMake runs on every build. Keeping the two apart means a public build never
needs a CUDA toolkit.

    python cpp/tools/prepare_cubins.py --sm 80 --sm 90

Every kernel is compiled once per (SM, dtype) it ships for — the *shipping
matrix*. AOT means no JIT fallback on the consumer's machine, so a combination
that is not enumerated here simply does not exist for a public user. Growing the
matrix is the cost of withholding the sources.

Regenerating is deliberate, not automatic: a rebuild rewrites large binaries in
git history, so it should be a reviewed change rather than a side effect.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]

# Import the kernel sources from the checkout, not from an installed package.
# The ordering forces it: building the extension needs the CUBINs, so this has to
# run before `pip install -e .`, which means `torch_cuda` is not importable yet.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
OUTPUT_DIR = ROOT_DIR / "cpp" / "kernels" / "vector_add" / "cubins"

# The CUBIN symbol `cuModuleGetFunction` looks up, and the `@cute.jit` entry
# point that emits it.
KERNEL_NAME = "vector_add_kernel"
ENTRY_POINT = "torch_cuda.dsl_kernels.cute.sm80_vector_add:vector_add"

DEFAULT_SMS = (80, 90)

# CuTeDSL target strings are not simply f"sm_{sm}": Hopper and Blackwell need the
# accelerated `a` variants, so the mapping is explicit.
GPU_ARCH_BY_SM = {80: "sm_80", 86: "sm_86", 89: "sm_89", 90: "sm_90a", 100: "sm_100a"}

# The sandbox ships one dtype. A real matrix keys the lookup on dtype as well,
# which means widening `find_embedded_cubin` in cubin_runtime.h to match.
DTYPE = "fp32"


@dataclass(frozen=True)
class CompileJob:
    """One cell of the shipping matrix."""

    sm: int
    dtype: str

    @property
    def target_arch(self) -> str:
        try:
            return GPU_ARCH_BY_SM[self.sm]
        except KeyError:
            raise ValueError(f"no CuTeDSL target architecture registered for SM{self.sm}") from None

    @property
    def filename(self) -> str:
        """Name `embed_cubins.py` parses back into (kernel, sm, dtype)."""
        return f"{KERNEL_NAME}.sm{self.sm}.{self.dtype}.cubin"


def compile_job(job: CompileJob) -> bytes:
    """Trace the kernel over fake tensors and return its SASS.

    Fake tensors carry dtype, rank, stride and alignment but no memory, so this
    compiles on a machine with a toolkit but no GPU. `cute.sym_int()` leaves the
    length dynamic; anything passed concretely would be baked in and would need
    its own matrix cell.
    """
    import cutlass  # noqa: F401  (registers the dtypes resolved below)
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    entry = _resolve_entry_point()
    ct_dtype = _cutlass_dtype(job.dtype)
    operand = make_fake_tensor(ct_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)

    with tempfile.TemporaryDirectory() as dump_dir:
        executable = cute.compile(
            entry,
            operand,
            operand,
            operand,
            make_fake_stream(),
            # CuTeDSL's option parser accepts --gpu-arch, not --arch; the full set is
            # --opt-level, --enable-assertions, --link-libraries, --generate-line-info,
            # --keep-cubin, --keep-ptx, --ptxas-options, --gpu-arch, --enable-tvm-ffi,
            # --dump-dir. An unknown flag fails the whole compile.
            options=f"--keep-cubin --dump-dir {dump_dir} --gpu-arch {job.target_arch}",
        )
        return _extract_cubin(executable)


def _resolve_entry_point() -> Any:
    import importlib

    module_path, _, attribute = ENTRY_POINT.partition(":")
    return getattr(importlib.import_module(module_path), attribute)


def _cutlass_dtype(name: str) -> Any:
    import cutlass

    try:
        return {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16, "fp32": cutlass.Float32}[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}") from None


def _extract_cubin(executable: Any) -> bytes:
    """`__cubin__` is a path on some CuTeDSL versions and raw bytes on others."""
    cubin = executable.__cubin__
    if isinstance(cubin, str):
        return Path(cubin).read_bytes()
    if isinstance(cubin, bytes | bytearray):
        return bytes(cubin)
    raise TypeError(f"unexpected __cubin__ type {type(cubin).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sm",
        type=int,
        action="append",
        dest="sms",
        help=f"target integer SM, e.g. 80 or 90; repeatable. Default: {' '.join(map(str, DEFAULT_SMS))}",
    )
    args = parser.parse_args()

    jobs = [CompileJob(sm=sm, dtype=DTYPE) for sm in (args.sms or list(DEFAULT_SMS))]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for job in jobs:
        payload = compile_job(job)
        destination = OUTPUT_DIR / job.filename
        destination.write_bytes(payload)
        total += len(payload)
        print(f"wrote {destination.relative_to(ROOT_DIR)} ({len(payload)} bytes)")

    print(f"{len(jobs)} images, {total} bytes of SASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# outbound coverage: touched so every published path has a real last-commit message
