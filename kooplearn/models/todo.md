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
### Changelog
1. `Datasets` files moved to `data/utils`

### Notes/questions for Grégoire:
1. (done) The `keops` backend has been dropped. You should remove every reference to it, starting from the function `BaseKoopmanEstimator._check_backend_solver_compatibility()` and backwards to every estimator you implemented.
3. (done) The `phi_testX` variable in `apply_eigfun` (direct estimators) is expecting the array $\phi(X_{{\rm test}})$.
5. (done) In the Kernel estimators it is implemented the `forecast` method, while in the Direct estimators the `predict` method. Let's make the interface uniform. I propose to stick with the `predict` name. Moreover, if possible, the `predict` functions in both Kernel and Direct methods should have the same signature. For example: the `which` argument in the Kernel estimator should be removed.
5. (done) In the `predict` function the `observable` variable should be an array of the observable evaluated at the Y training points, and not a lambda function as it is now. The rationale is that you might want to forecast an observable for which you have measurements, but not a closed-form expression.
2. The direct estimators are assuming that the arrays `X` and `Y` used in the fit function are already featurized (i.e. they are $\phi(X)$ and $\phi(Y)$, respectively). This is somehow in contrast with the Kernelized estimators where you need to pass a kernel explicitly to the class. Let's figure out together what is the best solution.
4. In the function `_get_cov`, are you sure that calling `np.cov(X.T, Y.T)` actually gives you the cross-covariance as expected? Please check!
6. The `predict` function in the Direct estimators is not implemented correctly. In particular, in each of the functions implemented in `_algorithms`, whenever you see a variable containing the string `phi_`, it is expecting the results of applying the feature map to an array (usually specified at the end of the variable name).
7. I have moved the tests in an outside folder.
8. As we are copy-pasting some old code, please check that the documentaiton is consistent and does not report wrong/unused variable names.

### Additional notes by Grégoire:
1. maybe the predict function should accept both behaviours (arrays of observables and functions), but it might be a bit ugly.
