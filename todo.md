### Algorithms
- [ ] Implement the primal algorithm to fit RRR without Tikhonov reg.
- [ ] Add the possibility to compute left eigenvectors for the primal algorithms.
- [ ] Implement `fit_rand_tikhonov` in primal.py
- [ ] Implement the following metrics:
    - [ ] Squared loss
    - [ ] Prediction error
- [ ] The `ScalarProduct` kernels are not interfaced with `FeatureMap`. Fix this.

### Data generation/Examples
- [ ] Add Brunton's method to the `encoder_decoder` models.
- [X] Finish the implementation of the triple well.
    - [x] Define the force function.
    - [ ] Add the eigenvalue decomposition.
        - [ ] Added from reference calculations, the domain sample is fixed. Consider whether to use `scipy`'s interpolation algorithms to make it callable at arbitrary points.
- [ ] Muller-Brown should be integrated without any dependencies.
- [ ] Add reference Koopman eigenvalues on stochastic problems.
    - [ ] Linear
    - [X] Logistic
    - [x] 1D triple well
    - [ ] Muller Brown
- [X] Duffing Oscillator
- [ ] Langevin driven by $\alpha$-stable noise.
- [ ] Add or check that is added a RNG for every stochastic term for reproducibility.
- [ ] Can be cool to add fluid-dynamics simulation. See, e.g. [JAX-FLUIDS](https://github.com/tumaer/JAXFLUIDS/) for an easy way to generate them.
### Testing
- [ ] Document functions
- [x] Test primal-dual matching on eigenfunctions (on top of eigenvalues). In general, study how coverage works.
- Test data generation for each one of the methods
    - [ ] Duffing
    - [ ] Lorenz63
    - [ ] Linear
    - [ ] Logistic
    - [ ] Muller-Brown
    - [ ] 1D Triple Well
- [ ] Test _randomized_ algorithms (not clear how to do that now).
- [ ] Test `nystrom` estimators (not clear how to do that).
- [x] Test the left eigenfunctions of the primal algorithm: FAILING.
- [ ] Add tests for the `ScalarProduct` abstract class and `TorchScalarProduct` class.
- [x] Add pre-computed assets for the eigenfunctions/eigenvalues of the 1D triple well as well as Muller-Brown if possible.
- [x] Handle the case of 0 Tikhonov regularization.

### Make the code clearer
- [ ] [Most important in my opinion] create and DOCUMENT a standard notation of the variables used throughout the 
  code, with name, shape and meaning of each variable.
- [ ] [Most important in my opinion] _src/operator_regression/dual -> Make the code clearer by providing references to 
  algorithms, specifying the equation that is being solved and occasionally commenting equations when many steps are 
  involved.
- [ ] [Most important in my opinion] _src/operator_regression/primal -> same as above
- [ ] models/edmd -> docstring + define every variable stored in the model (self + ['U_', 'V_', 'K_X_', 'K_YX_', 
  'X_fit_', 'Y_fit_'], define shapes of the variables
- [ ] models/kernel -> define every variable stored in the model (self + ['U_', 'V_', 'K_X_', 'K_YX_', 
  'X_fit_', 'Y_fit_'], define shapes of the variables
- [ ] models/kernel -> we are adding many hyperparameters for each solver, maybe one class for each solver or create 
  a solver object that we can pass for each model
- [ ] [Really Personal Opinion] I would avoid using any one-letter variable name, even if the context is clear. 
  For example, instead of C_X I would at least write cov_X as cov is the name of a common function from numpy 
  and other libraries. 