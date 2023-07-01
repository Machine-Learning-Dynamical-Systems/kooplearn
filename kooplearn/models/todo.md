- [x] Implement the `arnoldi` solver for primal algorithms
- [ ] Implement the primal algorithm to fit RRR without tikhonov reg.
- [ ] Add the possibility to compute left eigenvectors for the primal algorithms.
- [x] Implement the prediction algorithm for the primal problem
- [x] Incapsulate every algorithm into `sklearn` estimators with proper checks etc
- [x] Implement methods to compute the `svd` for both primal and dual algorithms
- [ ] Document functions
- [ ] Implement the following metrics:
    - [ ] Squared loss
    - [ ] Prediction error
- [ ] Add tests for each of the algorithm
- [ ] Primal and dual should give the same results on the same data (e.g. linear kernel)
- [ ] Implement fit_rand_tikhonov in primal.py
- [x] Complete eigfun in DirectEstimators.py (what is phi_testX? should we apply a feature map? If so, isn't it something at the general class level?)
- [ ] The `ScalarProduct` kernels are not interfaced with `FeatureMap`. Fix this.
- [X] In the Kernel estimators, double check that I need $K_{YX}$ and not its transpose.
- [ ] Add Brunton's method to the `encoder_decoder` models.
- [ ] Test primal-dual matching on eigenfunctions (on top of eigenvalues).
- [ ] Add 
### Changelog
1. `Datasets` files moved to `data/utils`
2. Added random number generator seed to the randomized algos (for my mental sanity).
3. Added some consistency tests for `primal/dual` algorithms. Skipping the tests of the `randomized` algorithms, as unclear what assertion to ask.
4. Added tests for Kernel estimators.

### Additional notes by Grégoire:
1. maybe the predict function should accept both behaviours (arrays of observables and functions), but it might be a bit ugly.
