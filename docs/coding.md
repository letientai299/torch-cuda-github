# Coding rules

The short version of what the hooks enforce, plus the parts they cannot.

## Enforced by the hooks

Run `prek install -t pre-commit -t commit-msg` once. Both CI pipelines run the
same [`prek.toml`](../prek.toml) hook set, so local and CI cannot disagree.

- Python is formatted and linted by [ruff][ruff]; the rule selection lives in
  `pyproject.toml`. The formatter owns line wrapping.
- C/C++/CUDA is formatted by `clang-format` against `.clang-format`.
- Markdown is linted by [rumdl][rumdl]; shell by [shellcheck][shellcheck] and
  formatted by [shfmt][shfmt].
- Every source file carries the two-line SPDX header; `insert-license` adds it
  where missing.
- Commit subjects follow [Conventional Commits][cc], checked by `cz check`.

Two trees are exempt, for reasons that are not style preferences:

- Binary artifacts — `.cubin` SASS and `.pt` goldens, both LFS-tracked. There is
  nothing to format, and the byte-array header built from them lives in the build
  tree, never in the repo.
- `torch_cuda/dsl_kernels/cute/` — the DSL resolves names through its own tracing
  machinery, so ruff reports undefined names that are not.

## Not enforced

- **Never let a backend silently degrade.** An explicitly requested backend that
  is unavailable raises. Silent fallback is how a packaging failure gets
  diagnosed for a week as a kernel regression.
- **Keep kernel sources out of the import path of `ops/`.** The ops import
  `dsl_kernels` lazily and only on the branch that needs it, so the published
  package stays importable with `cute/` filtered out. A top-level import there
  breaks the mirror and passes every internal test.
- **A constant shared across the build boundary gets a comment naming its twin.**
  `THREADS_PER_BLOCK` exists in the CuTeDSL source and again in the C++ launcher;
  the CUBIN bakes it in, so a silent divergence is a wrong answer, not a crash.
- **Every op keeps an eager reference implementation.** It is the oracle the
  tests compare against and the fallback that makes the package installable
  without a GPU.

[cc]: https://www.conventionalcommits.org/
[ruff]: https://docs.astral.sh/ruff/
[rumdl]: https://github.com/rvben/rumdl
[shellcheck]: https://www.shellcheck.net/
[shfmt]: https://github.com/mvdan/sh

## Sync test MR1

One commit, allowlisted path only, merged without squash.

### MR2 step 1

Third-commit squash test, step 1.

### MR2 step 2

Third-commit squash test, step 2.

### MR2 step 3

Third-commit squash test, step 3.

<!-- outbound coverage: touched so every published path has a real last-commit message -->

<!-- outbound 7: ticket key in the published subject -->

<!-- i2: labelled but unapproved -->

<!-- i3: pushed after approval -->

<!-- i7: a second approved commit -->

<!-- o2 commit 1 -->

<!-- o2 commit 2 -->

<!-- o2 commit 3 -->
