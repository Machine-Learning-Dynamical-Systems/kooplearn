<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="logo-light.png">
      <img alt="kooplearn logo" width="60%" src="logo-light.png">
    </picture>
</p>

<a href="https://kooplearn.readthedocs.io/latest/"><img alt="Static Badge" src="https://img.shields.io/badge/Documentation-informational"></a>
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Machine-Learning-Dynamical-Systems/kooplearn/CI.yml)
![GitHub License](https://img.shields.io/github/license/Machine-Learning-Dynamical-Systems/kooplearn)


`kooplearn` is a Python library to learn evolution operators —  also known as _Koopman_ or _Transfer_ operators — from data. `kooplearn` models can:

1. Predict the evolution of states *and* observables.
2. Estimate the eigenvalues and eigenfunctions of the learned evolution operators.
3. Compute the [dynamic mode decomposition](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition) of states *and* observables.
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

## Contributing

We welcome contributions from the community! If you're interested in contributing to `kooplearn`, please follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/issue-number`.
4. Make your changes and commit them with descriptive commit messages.
5. Push your changes to your forked repository.
6. Create a pull request from your branch to the `main` branch of the original repository.
7. Provide a clear title and description for your pull request, including any relevant information about the changes you've made.

We appreciate your contributions and will review your pull request as soon as possible. Thank you for helping improve `kooplearn`!

## License

This project is licensed under the [MIT License](LICENSE).

## Main contributors

`kooplearn` is an joint effort between teams at the Italian Institute of Technology in Genoa and the École polytechnique in Paris. The main contributors to the project are (in alphabetical order):
   - Vladimir Kostic
   - Karim Lounici
   - Giacomo Meanti
   - Erfan Mirzaei
   - Pietro Novelli
   - Daniel Ordonez
   - Grégoire Pacreau
   - Massimiliano Pontil
   - Giacomo Turri
   
The mantainer of this repo is Pietro Novelli.

## Citing `kooplearn`

```bibtex
@article{kooplearn,
title={kooplearn: A Scikit-Learn Compatible Library of Algorithms for Evolution Operator Learning}, 
author={Giacomo Turri and Grégoire Pacreau and Giacomo Meanti and Timothée Devergne and Daniel Ordonez and Erfan Mirzaei and Bruno Belucci and Karim Lounici and Vladimir R. Kostic and Massimiliano Pontil and Pietro Novelli},
year={2026},
eprint={2512.21409},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2512.21409}, 
}
```

---

We hope you find `kooplearn` useful for your dynamical systems analysis. If you encounter any issues or have suggestions for improvements, please don't hesitate to [raise an issue](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn/issues). Happy coding!

