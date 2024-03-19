> Do NOT delete completed tasks.

## General
- [ ] Do a complete pass on the documentation adding cross-links to class and attributes.
- [ ] Rename Least Squares models from `*DMD` to `LeastSquares` and update docs accordingly

## Documentation
- [ ] Add a _Operators & Dynamical Systems_ documentation page replacing the blog post, for now.
- [x] Add a Shared Bibliography file
- [x] Document everything in `kooplearn.abc`.
### Examples (tentative titles):
> Add name of the author and link to its social media for each of the examples.
- [ ] Large scale Kernel models with Nyström or Randomized techniques. (NoisyLogistic) - Giacomo
- [ ] Ordered MNIST (All algorithms) - Pie
- [ ] Sea temperature with Autoencoders (re-implementing Azencot's example) - Pie
- [ ] ALA2 (VAMPNets, DPNets) - Pie
- [ ] Regime switching - Grég

## Models
### NystromKDMD
- [ ] Fix the low level algorithms for numerical stability & test
- [ ] Use `kooplearn.models.Kernel` as an example to implement the Nystrom version. ~~Link to Falkon for a GPU implementation.~~
- [ ] Check the role of Tikhnonv reg. in both PCR and RRR -- [Giacomo's implementation](https://github.com/Giodiro/NystromKoopman/tree/9463faf5dd6a7b7a5ccba31ebbd755fcfa91e20f/nyskoop/estimators) is not 100% consistent (PCR has no reg, while RRR has it) 

### Nonlinear
- [x] Add `risk` method.
- [ ] Add power iteration QR normalization for the randomized RRR algorithms.
- [x] Lift the constraint of having 2D observables in `modes` and `predict`
- [x] Do as above for `Kernel`

### ~~EncoderModel~~ DeepEDMD Model
- [x] Rewrite it by subclassing `kooplearn.models.Nonlinear`
- [ ] Implement the loading and saving utilities - pay attention to saving the feature map.
- [x] Drop the requirements for the data to be of shape `[n_samples, n_features]`, and allow for a general `[n_samples, ...]`.

### Optimization
- [ ] Rewrite `kooplearn._src.operator_regression` in C++/Cython

### DPNets
- [x] Modify the `training_step` to get data from a context window.
- [ ] Add Chapman-Kolmogorov Regularization
- [x] Perform shape checks at the beginning of every epoch (`batch_idx == 0`).
- [x] Add shape checks: outputs should be 2D, and context windows should be two-dimensional. _Non-2D inputs are handled as  in_ `Nonlinear`
- [ ] Add a test in `tests`

### VAMPNets
- [x] Add shape checks: outputs should be 2D, and context windows should be two-dimensional. _Non-2D inputs are handled as  in_ `Nonlinear`
- [ ] Add a test in `tests`

### AutoEncoders
- [ ] Finish the `modes` and `eig` methods.
- [ ] Add  a test in `tests`

### Input-Output utilities
- [x] Add Numpy utilities to convert from a trajectory to a context window **view**. This avoids unnecessary memory usage. 
- [ ] Write a function to add `nan` padding for inference data (in which we do not know the target, nor the lookforward.)
- [x] Adapt Bruno's `datamodule`.
- [ ] Test saving and loading of every deep learning model

### Datasets
- [ ] The return of `Dataset.generate` should be a trajectory (as it is now, but double check).

### Refactoring to the context-window data paradigm

- [x] Take a decision on the name of the variables: current proposal is `data/contexts`, `lookback_len`.
- [x] Take a decision on the defaults of `lookback_len`. Either `lookback_len = 1` or `lookback_len = None`, that is taking in the context window _except the last snapshot_ as lookback. In practical scenarios I argue that the second option is better.
- [x] The lookback length should be defined a the model initialization, and not upon fitting.
- [x] The handling of custom observables is awkward, and inconsistent with the handling of state prediction/mode decomposition. Fix it. _Done: now only callable or_ `None` are accepted.

###### Module `kooplearn.abc`
- [x] Edit the Abstract Base Class definition for `kooplearn.abc.BaseModel` on `fit`, `predict`, `eig`, `modes`.
- [x] The `FeatureMap.cov` must be popped out of it and moved into `kooplearn._src.utils`, as I do not find a reasonable default behaviour in the context-window data paradigm.
- [x] `kooplearn.abc.TrainableFeatureMap` must now have a `lookback_len` keyword ~~(or equivalent, should we opt for a different nomenclature)~~.

###### Modules ~~`kooplearn.models.Nonlinear`~~, `kooplearn.models.Kernel` and `kooplearn.models.DeepEDMD`
- [x] Implement functions to be called at the I/O boundaries to reshape the I/O data in a coherent fashion.
- [x] `fit(X, Y) -> fit(data, lookback_len = None)`
- [x] Throw a ~~Warning~~ Error if `lookback_len != context_len - 1` (we cannot use future steps in these methods)
- [x] Save `lookback_len` at fitting so that it can be used back in prediction.
- [x] Implement the changes in the `fit` documentation.
- [x] `predict(X, t, observables)` should now return only the `lookback_len` prediction.
- [x] Observables, if passed as a precomputed array, must now be of shape `[n_init, context_len, ...]`. And they should be evaluated on the train dataset. We will perform the required slicing internally.
- [x] Assert that either `X.shape[1] == lookback_len`, or `X.shape[1] == context_len`. If `np.isnan(X[:, lookback_len:, :]).all() == False` throw an error. I could in principle raise a warning and discard the unused columns, but I prefer to be explicit to avoid confusion.
- [x] Document these changes accordingly.
- [x] Do for `modes(X, observables)` the same as done for `predict`.

###### Tests
- [x] Update `test_edmd_estimators`, `test_kernel_estimators`.
- [x] Add tests for the I/O utilities

### ~~DPNets~~ | LEGACY |
- Implement different regularization functions:
  - [x] Frobenius.
  - [x] Von-Neumann divergence.
  - [x] Log + Frobenius defined as $-\log(x) + x^2 - x$.
- [x] Remove the constraint for the data to be a dict with keys `x_value` and `y_value`.
- [x] ~~Design a flexible way to include different data timesteps in each batch, to then work with the Chapman-Kolmogorov regularization~~ See [the list fo the new data paradigm.](#refactoring-to-the-context-window-data-paradigm).
- [ ] Implement Saving and Loading functions to be called by `DeepEDMD`.

A point to reason on:
1. Each `TrainableFeatureMap` should come with its data-loading scheme, and should be able to produce the covariances and data arrays needed by the primal algorithms to work. The `predict` and `modes` functions should then work coherently with this data-loading structure.

Thoughts: this scheme might be a bit too general, and possibly detrimental. At this stage we only have DPNets to work with.

### Algorithms
- [ ] Implement the following metrics:
    - [x] Squared loss
    - [ ] Prediction error
    - [ ] Directed Hausdorff for spectra error estimation.
- [ ] Replace dynamic list creation (append) followed by torch.cat or torch.stack with an initialized tensor and 
  indexing (probably faster).

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
- [ ] Add or check that is added a RNG for every stochastic term for reproducibility.
- [ ] Can be cool to add fluid-dynamics simulation. See, e.g. [JAX-FLUIDS](https://github.com/tumaer/JAXFLUIDS/) for an easy way to generate them.
- [ ] Add pre-computed assets for the eigenfunctions/eigenvalues of the 1D triple well as well as Muller-Brown if possible.

### Testing
- [x] `low_level_primal_dual_consistency` is failing on the RRR algorithm. I have nailed down the fact that **in theory**, the non-zero eigenvalues of $K_{Y}$ and $C^{\dagger/2}_{X}C_{XY}C_{YX}C^{\dagger/2}_{X}$ should be the same, but in practice they are not.
- [x] Test primal-dual matching on eigenfunctions (on top of eigenvalues). In general, study how coverage works.
- [ ] Test data generation for each one of the methods
    - [ ] Duffing
    - [ ] Lorenz63
    - [ ] Linear
    - [ ] Logistic
    - [ ] Muller-Brown
    - [ ] 1D Triple Well
- [ ] Test _randomized_ algorithms (not clear how to do that now).
- [x] Test the left eigenfunctions of the primal algorithm.
- [x] Handle the case of 0 Tikhonov (ridge) regularization.
- [ ] *Sep 11, 2023:* Review the current tests and plan a comprehensive test suite for DMD-based estimators. 

### Make the code clearer
- [x] Create and DOCUMENT a standard notation of the variables used throughout the 
  code, with name, shape and meaning of each variable.
- [ ] `kooplearn._src.operator_regression.dual`: Make the code clearer by providing references to algorithms, specifying the equation that is being solved and occasionally commenting equations when many steps are involved.
- [ ] `kooplearn._src.operator_regression.primal`: As above
- [x] `kooplearn.models.edmd`: Docstring + define every variable stored in the model (self + `['U_', 'V_', 'K_X_', 'K_YX_', 
  'X_fit_', 'Y_fit_']`, define shapes of the variables
- [x] `kooplearn.models.kernel`: Define every variable stored in the model (self + `['U_', 'V_', 'K_X_', 'K_YX_', 
  'X_fit_', 'Y_fit_']`, define shapes of the variables
- [x] `kooplearn.models.kernel`: We are adding many hyperparameters for each solver, maybe one class for each solver or create 
  a solver object that we can pass for each model
- [x] I (Bruno) would avoid using any one-letter variable name, even if the context is clear. For example, instead of C_X I would at least write cov_X as cov is the name of a common function from numpy and other libraries. 