# torch-cuda

> **Throwaway sandbox.** Not a product, not maintained, not published anywhere.
> It exists to exercise a repository-sync workflow and will be deleted.

A deliberately small PyTorch + CUDA package. It exists to rehearse a
[Copybara][copybara] sync between a private GitLab repo and a public GitHub
mirror using a representative technology mix: PyTorch ops in front,
[CuTeDSL][cutedsl] and [Triton][triton] kernels behind, and a
[nanobind][nanobind] extension that carries ahead-of-time compiled CUBINs.

The point is the plumbing, not the math — every op is elementwise and has an
eager reference.

## Install

```sh
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

The kernel extension is opt-in and needs a CUDA toolkit:

```sh
python cpp/tools/prepare_cubins.py    # compile the CuTeDSL sources to SASS
TORCH_CUDA_BUILD_KERNELS=1 pip install -e .
```

Without it the ops fall back to Triton (with a GPU) or to eager PyTorch, so the
package installs and its tests pass on a laptop.

## Use

```python
import torch
from torch_cuda import vector_add

out = vector_add(torch.randn(1024), torch.randn(1024))
```

Which implementation ran is a single resolved value; see
[`torch_cuda/runtime/backend.py`](torch_cuda/runtime/backend.py). Force one with
the `backend=` argument or `TORCH_CUDA_BACKEND=eager|triton|cubin`.

## Layout

`torch_cuda/` is the Python package — `ops/` is the API, `runtime/` resolves
which implementation runs, and `dsl_kernels/triton/` holds the Triton kernels.
`cpp/` is the CMake build, the nanobind extension, and the compiled CUBINs it
loads. Tests are in `tests/`, examples in `examples/`.

This repository is a published mirror: development happens internally and the
public tree is a filtered copy, so some kernels ship compiled rather than as
source. [CONTRIBUTING.md](CONTRIBUTING.md) covers what that means for a pull
request.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/coding.md](docs/coding.md).
Style is enforced by [prek][prek]; install the hooks once with
`prek install -t pre-commit -t commit-msg`.

[copybara]: https://github.com/google/copybara
[cutedsl]: https://docs.nvidia.com/cutlass/
[nanobind]: https://nanobind.readthedocs.io/
[prek]: https://github.com/j178/prek
[triton]: https://triton-lang.org/

<!-- outbound coverage: touched so every published path has a real last-commit message -->
