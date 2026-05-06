# Contributing to kooplearn

Thank you for considering a contribution to `kooplearn`. Bug reports, documentation improvements, examples, tests, and code changes are all welcome.

## Development setup

Clone the repository and install the project from source with the optional runtime dependencies and development tools:

```bash
git clone https://github.com/Machine-Learning-Dynamical-Systems/kooplearn.git
cd kooplearn
uv sync --all-extras --all-groups
```

If you prefer `pip`, use an editable install. Dependency groups require `pip>=25.1`.

```bash
python -m pip install -U pip
python -m pip install -e ".[torch,jax]" --group dev --group docs --group examples
```

The core package does not require `torch` or `jax`. Install only the extras you need when working on a narrower change:

```bash
uv sync --extra torch --group dev
uv sync --extra jax --group dev
```

## Tests and linting

Run the default test suite with:

```bash
uv run pytest
```

Run the linter before opening a pull request:

```bash
ruff check
```

When changing optional PyTorch or JAX functionality, include the matching extra in your local environment and run the relevant tests:

```bash
uv run --extra torch pytest tests/test_torch_losses.py tests/test_feature_map_embedder.py
uv run --extra jax pytest tests/test_jax_losses.py tests/test_nnx_feature_map_embedder.py
```

## Reporting issues

When filing a bug report, include:

- The installed `kooplearn` version.
- Your Python version and operating system.
- The install command you used, including optional extras.
- A minimal code example or notebook cell that reproduces the problem.
- The full traceback or error message.

Feature requests should describe the use case, the current workaround if any, and the expected behavior.

## Pull requests

Please keep pull requests focused on one change. Include tests or documentation updates when the behavior changes, and mention any optional dependency needed to run the new code.

## Maintainer releases

The maintainer release process is documented separately in [RELEASE.md](RELEASE.md).
