(primer)=
# Operators and dynamical systems

> _Authors:_ Pietro Novelli — [@pie_novelli](https://twitter.com/pie_novelli) & Vladimir Kostic — [@vkostic30](https://twitter.com/vkostic30)

### Definitions
Given a _nonlinear_ dynamical system $x_{t + 1} = S(x_{t})$, the **Koopman operator** provides a _global linearization_ of the dynamics by mapping it to a suitable space of observables $\mathcal{F}$. An observable is any (scalar) function of the state. The Koopman operator $\mathsf{K}$ is defined as

$$(\mathsf{K}f)(x_{t}) = f(x_{t + 1}) := f \circ S (x_t) \qquad f \in \mathcal{F}.$$

Similarly, given a stochastic process $X:= \{ X_{s} \colon s \in \mathbb{N}\}$, its **Transfer operator** returns the expected value of any observable forward in time. The Transfer operator $\mathsf{T}$ is defined as

$$(\mathsf{T}f)(x) := \mathbb{E}\left[f(X_{t + 1}) \mid X_{t} = x \right] \qquad f \in \mathcal{F}.$$