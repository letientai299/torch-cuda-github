# AGENTS.md

## CRITICAL (YOU MUST)

- **Read and follow [`docs/coding.md`][coding]** before changing code.
- **Run the hooks before committing.** Install once with
  `prek install -t pre-commit -t commit-msg`; they then run on `git commit`. Or
  run ad hoc with `prek run`. Both CI pipelines enforce the same set.
- **MR/PR titles** start with a ticket key, then a Conventional-Commit summary,
  e.g. `[TCUDA-xxx] feat: ...`. Types: `feat`, `fix`, `docs`, `style`,
  `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- **Commit titles** follow the same shape without the required ticket key.
- **License headers.** Every source file carries the two-line SPDX header;
  `insert-license` adds it where missing.

## What this repo is

A sandbox that mirrors the production stack (PyTorch ops, CuTeDSL and Triton
kernels, a nanobind extension carrying pre-compiled CUBINs) at the smallest size
that still exercises the GitLab -> GitHub publish. Correctness of the math is
not the point; the publish boundary is. Prefer a change that makes a sync
scenario testable over one that makes an op faster.

[coding]: docs/coding.md
