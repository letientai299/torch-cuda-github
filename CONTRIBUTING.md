# Contributing

## How a change reaches this repository

Development happens in an internal repository, and this one is published from
it. Your pull request is reviewed here, then **imported** and merged internally,
and the merged change is replayed back onto this mirror.

The practical consequences, all of them worth knowing before you start:

- A maintainer queues the import once your pull request is approved and the
  checks are green. You do not need to ask for it in a comment. Queuing is
  visible: your pull request gets an `import-approved` label and a comment
  saying so.
- **Queuing is the last automatic message you will get until the change lands.**
  The internal side cannot write to GitHub, so it cannot report progress here.
  Silence after the label means the change is still in the queue, not that
  something went wrong. If the import cannot proceed, a maintainer comments.
- The internal test suite runs against production hardware that the public CI
  here does not have. If it fails, a maintainer will tell you what failed.
- Pushing a new commit after the label is applied means the import waits for a
  fresh approval: approvals are pinned to the exact commit they reviewed.
- **Your pull request will be closed, not merged.** GitHub cannot show a merge
  that happened somewhere else. You stay the author of the commit, and a comment
  on your pull request will link the commit once it appears on `main`.
- Not everything in the internal repository is published here, and a pull
  request that reaches into a path this mirror does not carry cannot be
  imported. If that happens the import is refused and a maintainer will explain
  what to do — it is not something you can diagnose from here.

## Developer Certificate of Origin

Every commit must be signed off. `git commit -s` appends the trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

The sign-off certifies that you wrote the patch or otherwise have the right to
submit it under this project's licence. See [developercertificate.org][dco].

Every commit is checked, not just the last one, and the check is required before
a pull request can be imported. To sign off commits you have already made:

```sh
git rebase --signoff origin/main
```

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

The title survives the import and becomes the subject of the commit that lands,
so it must be a valid [Conventional Commit][cc]: `feat: `, `fix: `, `docs: `,
`style: `, `refactor: `, `perf: `, `test: `, `build: `, `ci: `, `chore: `,
`revert: `.

## What CI can and cannot check

GitHub runners here have no GPU, so the checks on your pull request cover the
style gate and the eager code path only. Anything touching a kernel is exercised
after the import, on internal hardware. Say so in the PR description when your
change is GPU-only, and include the numbers you measured.

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

<!-- o3 commit 1 -->

<!-- o3 commit 2 -->

<!-- o3 commit 3 -->
