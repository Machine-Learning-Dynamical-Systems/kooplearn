# Release process

`kooplearn` uses [Semantic Versioning](https://semver.org/). Releases are maintainer-only and are published to PyPI from tags matching `v*` through the GitHub Actions release workflow.

## Preconditions

- Start from a clean worktree.
- Run the release from `main`.
- Ensure the test suite is green before tagging a release.

## Release steps

Use the existing `just` recipe to bump the version, create the tag, and push the release:

```bash
just release patch
just release minor
just release major
```

If no bump is needed, run:

```bash
just release
```

The recipe updates `pyproject.toml` and `uv.lock`, creates a `v<version>` tag, and pushes both `main` and the tag.
