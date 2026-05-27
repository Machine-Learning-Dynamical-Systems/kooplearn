<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/_static/logo-light.png">
      <img alt="kooplearn logo" width="60%" src="docs/_static/logo-light.png">
    </picture>
</p>

[![Docs](https://readthedocs.org/projects/kooplearn/badge/?version=latest)](https://kooplearn.readthedocs.io/latest/)
[![CI](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/Machine-Learning-Dynamical-Systems/kooplearn/graph/badge.svg?branch=main)](https://codecov.io/gh/Machine-Learning-Dynamical-Systems/kooplearn)
[![PyPI version](https://img.shields.io/pypi/v/kooplearn.svg)](https://pypi.org/project/kooplearn/)
[![Python versions](https://img.shields.io/pypi/pyversions/kooplearn.svg)](https://pypi.org/project/kooplearn/)
[![License](https://img.shields.io/github/license/Machine-Learning-Dynamical-Systems/kooplearn)](LICENSE)


`kooplearn` is a Python library to learn evolution operators —  also known as _Koopman_ or _Transfer_ operators — from data. `kooplearn` models can:

1. Predict the evolution of states *and* observables.
2. Estimate the eigenvalues and eigenfunctions of the learned evolution operators.
3. Compute the [Dynamic Mode Decomposition](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition) of states *and* observables.
4. Learn neural-network representations $x_t \mapsto \varphi(x_t)$ for evolution operators.

## Why Choosing `kooplearn`?

1. It is easy to use and strictly adheres to the [scikit-learn API](https://scikit-learn.org/stable/api/index.html).
2. **Kernel estimators** are state-of-the-art:

   * `kooplearn` implements the *Reduced Rank Regressor* from [Kostic et al. 2022](https://arxiv.org/abs/2205.14027), which is [provably better](https://arxiv.org/abs/2302.02004) than the classical [kernel DMD](https://arxiv.org/abs/1411.2260) in estimating eigenvalues and eigenfunctions.
   * It also implements [Nyström estimators](https://arxiv.org/abs/2306.04520) and randomized estimators [randomized](https://arxiv.org/abs/2312.17348) for blazingly fast kernel learning.
3. Includes representation-learning losses (implemented both in Pytorch and JAX) to train neural-network Koopman embeddings.
4. Offers a collection of datasets for benchmarking evolution-operator learning algorithms.

## Installation

To install the core version of `kooplearn`:

### **pip**

```bash
pip install kooplearn
```

### **uv**

```bash
uv add kooplearn
```

To enable neural-network representations using `kooplearn.torch` or `kooplearn.jax`:

### **pip**

```bash
# Torch
pip install "kooplearn[torch]"
# JAX
pip install "kooplearn[jax]"
```

### **uv**

```bash
# Torch
uv add "kooplearn[torch]"
# JAX
uv add "kooplearn[jax]"
```

### From source

For development, clone the repository and install the package with all optional extras and dependency groups:

```bash
git clone https://github.com/Machine-Learning-Dynamical-Systems/kooplearn.git
cd kooplearn
uv sync --all-extras --all-groups
```

With `pip>=25.1`, the equivalent editable install is:

```bash
python -m pip install -U pip
python -m pip install -e ".[torch,jax]" --group dev --group docs --group examples
```

## Testing

Run the default test suite from the repository root with:

```bash
uv run pytest
```

After installing with `pip`, use:

```bash
python -m pytest
```

## Contributing

We welcome contributions from the community. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, issue reports, and pull request guidance.

## License

This project is licensed under the [MIT License](LICENSE).

## Main contributors

`kooplearn` is an joint effort between teams at the Italian Institute of Technology in Genoa and the École polytechnique in Paris. The main contributors to the project are (in alphabetical order):
   - Vladimir Kostic
   - Karim Lounici
   - Giacomo Meanti
   - Erfan Mirzaei
   - Pietro Novelli
   - Daniel Ordoñez-Apraez
   - Grégoire Pacreau
   - Massimiliano Pontil
   - Giacomo Turri
   
The mantainer of this repo is Pietro Novelli.

## Citing `kooplearn`

```bibtex
@article{kooplearn,
title={kooplearn: A scikit-learn compatible library of algorithms for evolution operator learning},
author={Giacomo Turri and Grégoire Pacreau and Giacomo Meanti and Timothée Devergne and Daniel Ordoñez-Apraez and Erfan Mirzaei and Bruno Belucci and Karim Lounici and Vladimir R. Kostic and Massimiliano Pontil and Pietro Novelli},
year={2026},
eprint={2512.21409},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2512.21409}, 
}
```

---

We hope you find `kooplearn` useful for your dynamical systems analysis. If you encounter any issues or have suggestions for improvements, please don't hesitate to [raise an issue](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn/issues). Happy coding!
