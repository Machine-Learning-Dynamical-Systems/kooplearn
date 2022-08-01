## A short overview of the novel experiments.
### Noisy logistic map & Lorenz63
The noisy logistic map is defined by the equation $x_{t + 1} = 4x_{t}(1 - x_{t}) + \xi_{t}$, where $\xi_{t}$ is an additive i.i.d. noise term. The Lorenz63 system is the discretization of the differential equation $\frac{dx}{dt} = \begin{bmatrix}\sigma(x_{2} - x_{1}) \\ x_{1}(\mu - x_{3}) \\ x_{1}x_{2} - \beta x_{3} \end{bmatrix}$, with $\sigma = 10$, $\mu = 28$ and $\beta = 8/3$.

<p align="center">
<img src="complementary_figures/sample_trajectoriesjpg" alt="Sample Trajectories" width="640"/>
</p>

We have trained the Reduced Rank and Principal Component regressors on these two systems. The training and test errors as a function of the number of training behave as

<p align="center">
<img src="complementary_figures/errors_vs_num_samplesjpg" alt="TraningTest errors" width="640"/>
</p>

Each experiment was repeated 100 times. For both systems the number of test samples is $50000$. Given the large number of test samples, the test error is a good proxy of the true risk. By subtracting train and test errors, therefore, we can empirically verify the uniform bound in Theorem 3 of the manuscript. Plotting this "excess risk" bound in log scale, it is evident that the decay is faster than $\propto n^{-1/2}$, as predicted from the theortical result.

<p align="center">
<img src="complementary_figures/excess_riskjpg" alt="Excess risk" width="640"/>
</p>

### We have also added an example from computational Chemistry. Please refer to the notebook in the examples/alanine_dipeptide folder. 