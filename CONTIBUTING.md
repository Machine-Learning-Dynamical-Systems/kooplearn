## Release Process (Maintainers Only)

We use [Semantic Versioning](https://semver.org/). Releases are automated via the `release.sh` script found in the root directory.

Ensure your git status is clean and you are on `main`, then run:

```bash
# Usage: ./release.sh <patch|minor|major|alpha|beta|stable>
./release.sh patch
```