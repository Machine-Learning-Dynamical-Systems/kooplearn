# Gemini Code Review Report

This report details inconsistencies, bugs, and other issues found in the `kooplearn` repository.

## `src/kooplearn/_linalg.py`

- **`spd_neg_pow` function:**
    - [x] The docstring refers to `A` but the parameter is `M`.
    - [x] The `strategy` parameter's accepted values ("trunc", "tikhonov") are not documented.
    - [x] The return value is not documented.
- **`covariance` function:**
    - [x] The function is missing a docstring.
    - [x] The return value is not documented.
- **`weighted_norm` function:**
    - [x] The type hint for the `M` parameter is `ndarray | None`, but the docstring says `ndarray or LinearOperator`. The usage of `.T` implies it should be `ndarray`. This should be clarified.
- **`eigh_rank_reveal` function:**
    - [x] The function is missing a docstring.
    - [x] The parameters `values`, `vectors`, `rank`, `rcond`, `ignore_warnings` are not documented.
    - [x] The return value is not documented.
    - [x] There is a typo in the warning message: "Discarted" should be "Discarded".

## `src/kooplearn/_utils.py`

- **`stable_topk` function:**
    - [x] The docstring for `ignore_warnings` contains a typo: "discarted" should be "discarded".
    - [x] The warning message contains a typo: "Discarted" should be "Discarded".
    - [x] The warning message suggests decreasing "k", but the parameter is named `k_max`.
- **`find_complex_conjugates` function:**
    - [x] The type hint `np.ndarray[np.complexfloating]` is not standard. It should be `np.ndarray`, and the complex nature of the array should be described in the docstring.
    - [x] The return type hint `tuple[np.ndarray[np.int64], np.ndarray[np.int64]]` is not standard. It should be `tuple[np.ndarray, np.ndarray]`.
- **`fuzzy_parse_complex` function:**
    - [x] The function is missing a docstring.
    - [x] The return value is not documented.
- **`row_col_from_condensed_index` function:**
    - [x] The function is missing a docstring.
    - [x] The parameters `d` and `index` are not documented.
    - [x] The return value is not documented.

## `src/kooplearn/datasets/_logistic_map.py`

- **`logistic_map` function:**
    - [x] The function is missing a docstring.
- **`noise_features` function:**
    - [x] The function is missing a docstring.
- **`transition_matrix` function:**
    - [x] The function is missing a docstring.
- **`TrigonometricNoise` class:**
    - [x] The class is missing a docstring.
- **`make_noise_rng` function:**
    - [x] The function is missing a docstring.
- **`step` function:**
    - [x] The function is missing a docstring.
- **`invariant_distribution` function:**
    - [x] The function is missing a docstring.
- **`equilibrium_density_ratio` function:**
    - [x] The function is missing a docstring.
- **`_eval_eigenfunctions` function:**
    - [x] The docstring for the `eigenvectors` parameter states the shape is `(num_basis, num_modes)`, but the code iterates as `for i, coeffs in enumerate(eigenvectors):`, which implies the shape is `(num_modes, num_basis)`. This is inconsistent.
    - [x] The variable `coeffs` is used in the loop but it is not defined. It seems it should be `eigenvectors[i]`.
- **`compute_logistic_map_eig` function:**
    - [x] The docstring claims the returned eigenvalues are "sorted by magnitude in ascending order", but the code does not perform any sorting.

## `src/kooplearn/datasets/_ordered_mnist.py`

- **`fetch_ordered_mnist` function:**
    - [x] The docstring for the `images` return value states that the array is of type `uint8`, but the function scales the data to `float64`. The docstring should be corrected.

## `src/kooplearn/datasets/_overdamped_langevin_generator.py`

- **`assemble_operators_1d` function:**
    - [x] The `gamma` and `sigma` parameters are not documented in the `Args` section of the docstring.
- **`_assemble_2d_vectorized` function:**
    - [x] The function is missing a docstring.
- **`_assemble_2d_kronecker` function:**
    - [x] The function is missing a docstring.
- **`compute_prinz_potential_eig` function:**
    - [x] The docstring for the `eigenvalues` return value states they are "sorted by magnitude in ascending order", but the code sorts by the real part of the eigenvalues. This is incorrect as the eigenvalues can be complex. `sort_indices_by_magnitude` should be used with `np.abs(eigvals)`.

## `src/kooplearn/datasets/_samples_generator.py`

- **`make_logistic_map` function:**
    - [x] The `noise` parameter is defined in the function signature but is not used within the function.
    - [x] The docstring refers to a non-existent function `compute_logistic_map_eigenfunctions`. This should probably be `compute_logistic_map_eig`.
- **`_make_noise_rng` function:**
    - [x] The `TrigonometricNoise` class is defined inside `_make_noise_rng`. It could be defined at the module level.
- **`make_prinz_potential` function:**
    - [x] The docstring for the `gamma` parameter states a default value of `0.1`, but the function signature defines it as `1.0`.

## `src/kooplearn/structs.py`

- **`FitResult` class:**
    - [x] The type hints for `U`, `V`, and `svals` use `NDArray[np.float64]`, which is not a standard type hint. They should be `np.ndarray`.
- **`DynamicalModes` class:**
    - [x] The `get_right_eigenfunction` method's docstring and type hint indicate a `complex` return type, but the implementation returns a `np.ndarray`. This is a bug.


## `src/kooplearn/metrics.py`

- **`directed_hausdorff_distance` function:**
    - [x] There is a typo in the docstring's code block directive: `.. code-block:python` should be `.. code-block:: python`.

## `src/kooplearn/jax/nn/_functional.py`

- **`vamp_loss` function:**
    - [x] The `x` and `y` parameters are not documented in the docstring.
- **`orthonormal_fro_reg` function:**
    - [x] The math formula in the docstring is not rendered correctly. It should be `\frac{1}{D} \| \mathbf{C}_{X} - I \|_F^2 + 2 \| \mathbb{E}_{X} x \|^2`.
- **`orthonormal_logfro_reg` function:**
    - [x] The math formula in the docstring is not rendered correctly. It should be `\frac{1}{D}\text{Tr}(C_X^{2} - C_X -\ln(C_X))`. 

## `src/kooplearn/torch/nn/_functional.py`

- **`orthonormal_logfro_reg` function:**
    - A `TODO` comment is present in the code, which should be addressed.

## `src/kooplearn/linear_model/_regressors.py`

- **`__all__` variable:**
    - [x] The `__all__` variable lists function names with a `primal_` prefix (e.g., `"primal_pcr"`), but the actual function names in the file do not have this prefix (e.g., `pcr`). This will cause import errors when using `from kooplearn.linear_model._regressors import *`.
- **`estimator_risk` function:**
    - [x] The docstrings for `cov_Xv`, `cov_Yv`, `cov_XYv`, and `cov_XY` describe shapes related to the number of samples (`n_val`, `n_train`), but these are covariance matrices and should have shapes related to the number of features.
- **Missing Docstrings:**
    - [ ] The following functions are missing docstrings:
        - `eig`
        - `evaluate_eigenfunction`
        - `svdvals`
        - `pcr`
        - `rand_pcr`
        - `_reduced_rank_noreg`
        - `reduced_rank`
        - `rand_reduced_rank`

## `src/kooplearn/kernel/_regressors.py`

- **`svdvals` function:**
    - [ ] The function is missing a docstring.
- **`rand_pcr` function:**
    - [ ] The function is missing a docstring.
- **`nystroem_reduced_rank` function:**
    - [ ] The line `+reg * kernel_X * (num_centers**-1)` seems to be a typo and has no effect.
- **`rand_reduced_rank` function:**
    - [ ] The `precomputed_cholesky` parameter is not documented in the docstring.

## `src/kooplearn/linear_model/_base.py`
    - [x] The docstring for the `_init_covs` method is incorrect. It describes a `stacked` parameter that the function does not accept. The function actually accepts `X` and `Y`.

## `src/kooplearn/kernel/_base.py`

- **`KernelRidge` class:**
    - [x] The docstring for the `optimal_sketching` parameter contains a typo: "computaitonally" should be "computationally".

## `src/kooplearn/kernel/_nystroem.py`

- **`NystroemKernelRidge` class:**
    - [x] The `_init_nys_kernels` method is missing a docstring.
    - [x] The `_center_selection` method is missing a docstring.
    - [x] The `__sklearn_tags__` method sets `non_deterministic = self.eigen_solver == "randomized"`, but `"randomized"` is not a valid option for the `eigen_solver` parameter in this class.