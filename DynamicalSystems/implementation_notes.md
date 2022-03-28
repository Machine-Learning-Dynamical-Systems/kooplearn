## To-do
1. Add history to sample and check performance of "autoregressive" models for forecasting
2. Implement mode decomposition and forecast for given observable $f$. 
3. Find low-rank dynamical system (syntetic) to compare with the full-rank decomposition.
## Implementation details: low rank projector:
The eigenvalue problem to solve is
$$ \frac{1}{n}K_{X}K_Yv_{i} = \sigma^{2}_{i}\left(K_{X} + n\eta \rm{Id}\right)v_{i},$$
In the unregulatised limit $\eta \to 0$ one has the simple EVD
$$\frac{1}{n}K_{Y}v_{i} = \sigma^{2}_{i}v_{i}.$$
In both cases, the projectors should be $n^{-1}K_{Y}$-orthonormal. It is however sufficient to take $V_{r} \in \mathbb{R}^{n\times r}$ such that $V_{r}^{*}K_{Y}V_{r} = \delta_{ij}$ and then return $\sqrt{n}V_{r}$.
### Eigenvalue problem - Regularized case:
Define $U :=  K_{XY}V_{r}$, and $T := K_{Y}V_{r}$. With Conjugate Gradient iterations get 
$$V := \left(K_{X} + n\eta \rm{Id}\right)^{-1}T.$$
### Eigenvalue problem - Unregularized case:
Define $U :=  K_{XY}V_{r}$. With Least Squares solver get 
$$V := K_{X}^{\dagger}V_{r}.$$ 

For both cases now solve solve the EVD 
$$U^{*}V w_{i} = \lambda_{i}w_{i}.$$
Finally return the eigenvectors matrix $W:= V[w_{1} | \ldots |w_{r}]$.

## Alternative implementation details: low rank projector:
The eigenvalue problem to solve is
$$ \frac{1}{n}K_{Y}K_{X}u_{i} = \sigma^{2}_{i}\left(K_{X} + n\eta \rm{Id}\right)u_{i},$$


The vectors $u_{i}$ should be normed so that $u_{i}^{{\rm T}}\left[n^{-1}K_{X}\left(K_{X} + n\eta \rm{Id}\right)\right]u_{i} = \sigma_{i}^{2}$. I then define $V_{r} := K_{X}U_{r}\Sigma^{-2}_{r}n^{-1/2}$.
### Eigenvalue problem
Now solve solve the EVD 
$$\frac{1}{n}V_{r}^{*}K_{YX}U_{r} w_{i} = \lambda_{i}w_{i}.$$
Finally return the eigenvectors matrix $W:= V[w_{1} | \ldots |w_{r}]$.

