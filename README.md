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

Where things live and, more importantly, which side of the publish boundary they
sit on: see [docs/publishing.md](docs/publishing.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/coding.md](docs/coding.md).
Style is enforced by [prek][prek]; install the hooks once with
`prek install -t pre-commit -t commit-msg`.

[copybara]: https://github.com/google/copybara
[cutedsl]: https://docs.nvidia.com/cutlass/
[nanobind]: https://nanobind.readthedocs.io/
[prek]: https://github.com/j178/prek
[triton]: https://triton-lang.org/
