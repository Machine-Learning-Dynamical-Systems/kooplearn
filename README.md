<p align = "left">
  <img src="logo.svg" alt="SVG Image" style="width:50%;"/>
</p>

# Learn Koopman and Transfer operators for Dynamical Systems and Stochastic Processes

`kooplearn` is a Python library designed for learning Koopman or Transfer operators associated with dynamical systems. Given a _nonlinear_ dynamical system $x_{t + 1} = S(x_{t})$, the **Koopman operator** provides a _global linearization_ of the dynamics by mapping it to a suitable space of observables $\mathcal{F}$. An observable is any (scalar) function of the state. The Koopman operator $\mathsf{K}$ is defined as $$(\mathsf{K}f)(x_{t}) = f(x_{t + 1}) := f \circ S (x_t) \qquad f \in \mathcal{F}.$$
Similarly, given a stochastic process $X:= \{ X_{s} \colon s \in \mathbb{N}\}$, its **Transfer operator* returns the expected value of any observable forward in time. The Transfer operator $\mathsf{T}$ is defined as $$(\mathsf{T}f)(x) := \mathbb{E}\left[f(X_{t + 1}) \mid X_{t} = x \right] \qquad f \in \mathcal{F}.$$

`kooplearn` provides a suite of algorithms for model training and analysis, enabling users to perform forecasting, spectral decomposition, and modal decomposition based on the learned operator.

Please note that `kooplearn` is currently under active development, and while we are continuously adding new features and improvements, some parts of the library might still be a work in progress.

## Features

- Implement different algorithms to learn Koopman or transfer operators for dynamical systems.
- Perform forecasting using the learned operators.
- Conduct spectral decomposition of the learned operator.
- Perform modal decomposition for further analysis.
  
## Installation
To install the core version of `kooplearn`, without optional dependencies, run
```bash
   pip install kooplearn
```
To install the full version of `kooplearn`, including Neural-Network models, and the dahsboard, run
```bash
   pip install "kooplearn[full]"
```
To install the development version of `kooplearn`, run
```bash
   pip install --upgrade git+https://github.com/Machine-Learning-Dynamical-Systems/kooplearn.git
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

---

We hope you find `kooplearn` useful for your dynamical systems analysis. If you encounter any issues or have suggestions for improvements, please don't hesitate to [raise an issue](https://github.com/Machine-Learning-Dynamical-Systems/kooplearn/issues). Happy coding!

