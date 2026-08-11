# Contributing

## Developer Certificate of Origin

Every commit must be signed off. `git commit -s` appends the trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

The sign-off certifies that you wrote the patch or otherwise have the right to
submit it under this project's licence. See [developercertificate.org][dco].

## Before you open a PR

```sh
prek install -t pre-commit -t commit-msg   # once
prek run --all-files
pytest
```

`prek` is the same gate CI runs. Fix what it reports rather than passing
`--no-verify`; a hook that is wrong is worth a PR against
[`prek.toml`](prek.toml).

## PR titles

The title becomes the squash-merge commit message, so it must be a valid
[Conventional Commit][cc]: `feat: `, `fix: `, `docs: `, `style: `, `refactor: `,
`perf: `, `test: `, `build: `, `ci: `, `chore: `, `revert: `.

## What CI can and cannot check

GitHub runners have no GPU. CI covers the style gate and the eager code path
only; anything touching a kernel is reviewed by hand and validated on internal
hardware. Say so in the PR description when your change is GPU-only, and include
the numbers you measured.

## Kernel changes

The CuTeDSL kernel sources are not part of this repository — the package ships
their compiled form. A PR that needs a kernel change should describe the required
behaviour and open an issue; a maintainer lands the kernel side and republishes
the compiled artifact. See [docs/coding.md](docs/coding.md).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

[cc]: https://www.conventionalcommits.org/
[dco]: https://developercertificate.org/

<!-- outbound coverage: touched so every published path has a real last-commit message -->
