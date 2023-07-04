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
    - [ ] Define the force function.
    - [ ] Add the eigenvalue decomposition.
- [ ] Muller-Brown should be integrated without any dependencies.
- [ ] Add reference Koopman eigenvalues on stochastic problems.
    - [ ] Linear
    - [X] Logistic
    - [ ] 1D triple well
    - [ ] Muller Brown
- [X] Duffing Oscillator
- [ ] Langevin driven by $\alpha$-stable noise.
- [ ] Add or check that is added a RNG for every stochastic term for reproducibility.
### Testing
- [ ] Document functions
- [ ] Test primal-dual matching on eigenfunctions (on top of eigenvalues). In general, study how coverage works.
- Test data generation for each one of the methods
    - [ ] Duffing
    - [ ] Lorenz63
    - [ ] Linear
    - [ ] Logistic
    - [ ] Muller-Brown
    - [ ] 1D Triple Well
- [ ] Test _randomized_ algorithms (not clear how to do that now).