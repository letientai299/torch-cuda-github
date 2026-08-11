# What is published, and what is not

This repo is the internal source of truth. A filtered copy is published to a
public GitHub mirror. The rule is an allowlist: a path that is not listed is not
published, so a new top-level directory is private by default.

## The boundary

| Tree                             | Published | Why                                                                             |
| -------------------------------- | --------- | ------------------------------------------------------------------------------- |
| `torch_cuda/ops/`                | yes       | The API consumers call                                                          |
| `torch_cuda/runtime/`            | yes       | Selects the backend; needed to interpret a bug report                           |
| `torch_cuda/dsl_kernels/triton/` | yes       | Generic fusions, no differentiating IP; JIT-compiled, so the source must ship   |
| `torch_cuda/dsl_kernels/cute/`   | no        | The kernel IP. Consumers get the compiled SASS instead                          |
| `cpp/kernels/*/cubins/*.cubin`   | yes       | The compiled SASS (LFS) — the artifact that replaces the withheld sources       |
| `cpp/` (rest)                    | yes       | Loader and bindings; without them the CUBINs are unusable                       |
| `.github/`                       | yes       | Only meaningful on the mirror, but authored here                                |
| `3rdparty/`                      | yes       | Public upstreams, published as gitlinks; consumers recurse from `.gitmodules`   |
| `.gitlab/`, `.gitlab-ci.yml`     | no        | Internal runners, registries, and credentials                                   |
| `docs/nv/`                       | no        | Internal notes; nested inside a published tree, so it needs an explicit exclude |
| `sync/`                          | no        | The publish tooling itself                                                      |

`docs/nv/` is the one case that needs an exclude entry rather than simply being
unlisted: everything else private is a top-level path the allowlist never names.

## The boundary runs through `dsl_kernels/`, not around it

`triton/` ships and `cute/` does not, yet both are "kernel sources" under the
same parent. Anything that globs `dsl_kernels/**` gets this wrong in one
direction or the other — either the Triton ops break for every public user, or
the CuTeDSL kernels leak. Both directories are covered by
[tests](../tests/test_ops.py) that must keep passing on the mirror.

The split is deliberate, on two grounds:

- **IP.** The Triton kernels are generic fusions — SwiGLU, LayerNorm + projection,
  axis moves and padding — the kind found in any public Triton tutorial. The
  differentiating work is the CuTeDSL attention and GEMM kernels. Publishing the
  first set gives away nothing the second set protects.
- **Cost.** Withholding a kernel means AOT-compiling it, and AOT means a shipping
  matrix: a JIT backend compiles whatever shape and dtype it is handed, an AOT one
  only has the variants someone enumerated. Triton's AOT path fixes dtypes,
  `constexpr` values, `num_warps`, and pointer-divisibility hints _per variant_,
  so its matrix is combinatorially larger than CuTeDSL's — which keeps lengths
  dynamic via `cute.sym_int()`. Paying that for kernels with no IP to protect buys
  nothing.

For the CuTeDSL half the matrix is real and load-bearing: a combination outside
[it](../cpp/tools/prepare_cubins.py) does not exist for a public user — an eager
fallback at best, an error at worst.

## The artifact is the file, not the header

The SASS ships as one `.cubin` per (kernel, SM, dtype), tracked in LFS. The
byte-array header that links them into the extension is generated on every build
by [`embed_cubins.py`](../cpp/tools/embed_cubins.py) and never committed — so
compiling kernels (needs a toolkit and the private sources) stays separate from
embedding them (needs neither). That separation is what lets a public clone build.

Two consequences the sync has to respect:

- **LFS objects are not ordinary git data.** The pointer file is versioned; the
  bytes live on a per-host LFS server. A publish that replays commits without
  pushing objects to the destination's LFS endpoint produces a mirror that looks
  right and is broken. [`tests/test_golden.py`](../tests/test_golden.py) is the
  canary — it fails loudly when it finds a pointer where contents should be.
- **No images is a valid state.** An unbuilt tree, or a clone without LFS, yields
  an empty lookup rather than a build failure, and the Python layer falls back.

## Checks

Three layers, in order:

1. The allowlist in the sync config — the only real control.
2. A secret scan over the produced tree, before anything is pushed.
3. [`.github/workflows/public-hygiene.yml`](../.github/workflows/public-hygiene.yml)
   on the mirror — a backstop that runs after publication. A finding there means
   layer 1 or 2 is broken.

## Round trip

Contributions arriving as GitHub PRs come back to this repo. A published commit
carries a `GitOrigin-RevId` trailer so re-running the outbound sync is
idempotent and does not replay what it already sent.
