# Roadmap
#### Mar 14, 2024:
**Implementations**.
1. Consistent AE. To be tested on the `ordered_MNIST` example.

**Documentation & Examples**.
1. Add docstrings (google style) to 
   - `kooplearn.abc.ContextWindow`
   - `kooplearn.abc.ContextWindowDataset`
   - `kooplearn.data.TensorContextDataset`
   - `kooplearn.data.TrajectoryContextDataset`
2. Update the guide in `docs/guides/data_specs.md` to introduce the new Context Windows classes and methods.
3. Create a jupyter notebook 'Using Context Windows', in which are illustrated the methods and attributes:
   - `lookback()`
   - `lookforward()`
   - `slice()`
   - `context_length`
   - `shape` (For Tensor Contexts)
   
   Show a practical example of `traj_to_contexts` with a simple `np.arange(20)`, and what is returned by the functions `predict`. To be clear: `predict` now returns a Tensor of the same shape of shape `(len(data), 1, data.shape[:2])`.
4. In the Ordered MNIST example, I have sketched a couple of cells to check whether the eigenfunctions of a model provide a good dimensionality reduction. I plotted the value of the eigenfunction in a 2d space for each image in the test set and colored with respect to the true label. We see that points with the same color (that is corresponding to the same labels) are clustered toghether. Can you write it down nicely? In this way we test both the forecasting and the eigenvalue decomposition for this example.


#### Mar 11, 2024:
Some notes on the AutoEncoder models:
1. The signature of `predict` should match `Nonlinear.predict` (see how the observables are changed). Same for `modes`, even if it is not implemented.
2. `eig` still accepts arrays rather than `ContextWindowDataset`.

In addition: 
1. I am working on `kooplearn.abc` to update the signatures of `BaseModule` too.
2. Should port some methods from `kooplearn._src.operator_regression.primal` to `torch`, to be used with the deep feature maps.

(LATER): Upon testing the ordered MNIST, some points are emerging:
1. Write a `collate_fn` to do batching from a `TensorContextDataset`.


#### Mar 10, 2024:
Make everything compatible with `torch.compile`.
#### Mar 9, 2024:
Some preliminary notes on the updates for the NN part. The main goal should be to have classes which are flexible and comprehensive. In this respect, using Lightning is good. 

Some actual implementation details:
1. We leave the Auto Encoder as now
2. We implement the `VAMP` and `DP` losses as `torch Modules` as done, e.g. in `torch.nn` (see the losses section)
3. We replace `kooplearn.models.feature_maps.DPNets` and `kooplearn.models.feature_maps.VAMPNets` by a single module which takes one of these losses defined in 2. as argument. It is just a matter of rewrapping some stuff, nothing big. 
4. In the spirit of simplifying I am also considering to remove `kooplearn.modules.DeepEDMD`, as it is ust a light wrapper around `kooplearn.modules.EDMD`.

#### Mar 6, 2024:
Defining the new context API.

**Base class**: `ContextWindow` should just be an Iterable, and implement
1. Attributes:
    - `context_length`
2. Methods:
    - `lookback(lookback_length: int, slide_by: int = 0)`
    - `lookforward(lookback_length: int)` 
    - `save`
    - `load`
    - `__repr__`
3. It should be a subclass of `collections.abc.Sequence`

**ContextWindowDataset**: A collection of context windows. Should be subclassed to be used both with `torch` or `numpy`. Should implement:
- `__init__`
- `__len__`
- `__iter__`

(**LATER**) To do for tomorrow:
1. Organize the `torch Dataset` classes for the new context windows.
2. Update the AutoEncoder models to deal with it.
3. Add `fsspec` to `pickle_load` and `pickle_save` to store artifacts on the cloud.
4. Review the whole documentation and port the examples in the new formalism.

#### Mar 5, 2024:
Features to add to `Contexts`:
- A bijective map of indices from the original trajectory (possibly, trajectories) to the element of the context window length.

#### Mar 4, 2024:
Added a branch `better_contexts`, to implement a better management of the data flow. The main goal is to have a more intuitive way to call the functions `modes`, `predict`, and `eig` (for the evaluation of the eigenfunctions) in each model. Upon merging to main, this will result in the release of a new minor version. Without overhauling the current design, we will keep the following principles:
1. `Contexts` only specify the context length, whereas the lookback window should be specified in each model.
2. Lookback windows can be used for inference, lookforward windows only for train.
3. Re-use the utilities in `kooplearn._src.operator_regression.utils` to do shape checks.
4. Implement a `torch` version of contexts to collate them easily even in the case of GNNs.
5. **Extra**: add a `reencode_every` keyword argument to `predict` as proposed in [Course Correcting Koopman Representations](https://arxiv.org/abs/2310.15386).

**Open questions.**

- The class should contain only 1 context or a batch of them? 

**Status.**

- Implemented an initial `Contexts` class. Add shape checks and remove the explicit definition of `lookback_length` as a property.
- Update every model to use `Contexts`. I should do minimal surgery, editing as close as possible to the API boundaries. I am just adding an abstraction.


#### Jan 24, 2024:
Testing the old and new versions of the algorithm. Some observations:
- At low (approaching machine precision) regularization strengths, or when the rank is too high, the computed estimator is highly unstable. As expected.
- The old function `_fit_reduced_rank_regression_noreg` seem to be extremely well conditioned. Might be worth to retain it. 
- Even with the numerical stabilization in place, if the rank is too high, the estimated models might differ.

#### Jan 23, 2024:
Working on RRR. Return by default the `svals_sq`, which are related to the spectral bias. Save it as as an attribute of the model. Extend the filtering mechanism to Rand-RRR and Nystroem-RRR. Add the possiblity to add a pre-computed cho_factor to the Rand-RRR for numerical efficiency. Use iterative methods otherwise.

#### Oct 17, 2023:
Added branch `in_out` in which I will work on serialization of each model, and more generally on every input-output utility which might be needed.

- The easiest storage option is just using `pickle`. `sklearn` as well as `lightning` models should be serializable by default.
- With [PEP 574](https://peps.python.org/pep-0574/), `pickle` introduced a 'protocol 5' to efficiently serialize large objects as Numpy Array.
- Implement a test for each serialization. 
- I should be able to save and load models and feature maps, as specified in `kooplearn.abc`.
- Add the possibility to pass both a path-like or a file-like object. The file-like object can be then used with `fsspec`.


#### Oct 3, 2023:
Macro implementations left to do:
- Nystrom Kernel methods (add a `NystromKernel` model)
- Test Randomized solvers.
- Implement `modes` and `eig` for AutoEncoders
- Implement dictionary learning schemes listed in the note of Sep 12 - 18
- Saving and loading for every model: implementation & test
- Documentation write up
- Example datasets (in or out of the library?!)
***
#### Oct 1, 2023:
Add github actions to build the documentation at every release. Plan for a rollout of `kooplearn 1.0.0` on PyPi. Start with github tagging.

Rationalize examples. 
- PEMS-Bay dataset
- Reproduce one representative example per method, taking it from the paper in which the method was introduced.
***
#### Sep 19, 2023:
Do **not** perform shape inference, on tensors of with `ndim` smaller than the minimum required. I should rather return a (standard) Error.

Rationalize checks and errors throughout the code.

Add context windows utilities for `torch`.

Create a common benchmark to run every model against. 

`.fit()` should ingest tensors of shape `[batch_size, context_len, *features]`, while `.predict()`, `.eig()` and `.modes()` should ingest tensors of shape `[batch_size, lookback_len, *features]`.

In the docs provide a table with the implemented models, references and notes.
***
#### Sep 12 - 18, 2023:
A list of models to implement:

- [x] [Deep learning for universal linear embeddings of nonlinear dynamics
 (2017)](https://arxiv.org/abs/1712.09707)
    - Not adding the sup-norm term for the moment. Ask the authors clarifications about it. 
    - Need to implement the `modes` and eigenfunction evaluation. (Done for this will be done for every AE model)
    - In the `kooplearn` data paradigm describe how the basic functions of `kooplearn.abc.BaseModel` should work in practice. At the moment the scheme is that `predict: [batch_size, context_len, *features] -> [batch_size, *features]`.  - _TO BE TESTED AND DOCUMENTED_
- [x] [Deep Dynamical Modeling and Control of
Unsteady Fluid Flows](https://arxiv.org/pdf/1805.07472.pdf) - _TO BE TESTED AND DOCUMENTED_
- [x] [Forecasting Sequential Data using Consistent Koopman Autoencoders - _TO BE TESTED AND DOCUMENTED_
 (2020)](https://arxiv.org/abs/2003.02236)
- [ ] [Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition
(2017)](https://arxiv.org/abs/1710.04340)
- [ ] [Learning Deep Neural Network Representations for Koopman Operators of Nonlinear Dynamical Systems
(2017)](https://arxiv.org/abs/1708.06850)
- [ ] [Linearly-Recurrent Autoencoder Networks for Learning Dynamics
(2017)](https://arxiv.org/abs/1712.01378)
- [ ] [Extended dynamic mode decomposition with dictionary learning: a data-driven adaptive spectral decomposition of the Koopman operator (2017)](https://arxiv.org/abs/1707.00225)
- [x] [VAMPnets for deep learning of molecular kinetics](https://www.nature.com/articles/s41467-017-02388-1)
  - Missing the $p \neq 2$ case to implement in `kooplearn.nn.functional`
  - ~~Missing docstrings~~
***
#### Sep 11, 2023:
Implemented most of the functionality and docs for the context window data paradigm. Still missing `DPNets` & other auto-encoder based methods.

Need a general strategy to test and sanitize the input algorithms. At the moment tests are sprinkled all over the code in an unorganized fashion.
***
#### Sep 4, 2023
###### Pie&Vladi:
Agreed on redefining the smallest unit of data accepted by all of our algorithms as a tuple: `(ArrayLike[context_window, features], target_idx)`.

<p align = "center">
  <img src="assets/context_window_cheme.svg" alt="SVG Image" style="width:50%;"/>
</p>

Above, pictorial representation of the context-window based data approach.
***
#### Aug 20, 2023
###### Pie:
For the kernel DMD I have dropped the custom kernel objects defined in `kooplearn._src.kernels` and relied instead on `sklearn.gaussian_processes.kernels`, which better documented, supported and allow for easy HP tuning. Minimal API changing, working to have running tests. Check that `Nonlinear` models are working too.
***
#### Jul 18 '23
###### Bruno:
- [x] Implement the [code for the paper](https://github.com/BethanyL/DeepKoopman) "Deep learning for universal linear embeddings of nonlinear dynamics" by Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton

###### Grégoire:
1. Finish visualization tools for modes

###### Pie:
1. Review Bruno's code for `dpnets`
    1. Add a `JAX` analogue model
    2. Test Bruno's implementation on an easy dataset
2. Everything from the [Jul 13](#jul-13-23) list
- [x] Add the possiblity to set a seed for the randomized algorithms in `kernel` and `edmd` models.

###### Vladi:
1. Impact of the pre-processing/windowing steps on the estimators
2. Study how the spectral filtering regularization schemes interact with the estimation of the eigenvalues **especially** if they are complex.

##### Features to add before private release:
- Visualization tools using `dash`.
- I/O utilities
    - Time-series dataloader for `torch`, `jax`, and `numpy`
    - Context windows, chunking of the trajectory
    - Adding time-related features
- Metrics
    - Reconstruction losses
    - Probabilistic losses
***
## Jul 13 '23
###### Pietro:
- [x] Add to the abstract `BaseModel` class two methods: `load` and `save` to save the estimators. Implement them for the `edmd` and `kernel` models.
- [x] Add a **robust** function that, given a list of complex numbers containing only pure reals and couples of complex conjugates, return two masks, one with the indices of the real numbers and one with the indices of only one of the elements of the CC pair.
   - Add complementary function to perform checks and sum reductions
- [x] Modify the mode function to return **everything** (i.e. including the initial condition) except of the eigenvalue.
***
## Jul 3 '23
###### Grégoire:
- [x] Add few synthetic financial datasets
- [x] Add a minimal implementation of the viz functionality (`matplotlib` for the moment. Look into `dask` and `gradio`)

###### Bruno:
1. Implement a functioning `DPnets` feature map
2. Implement utility functions to: data loading (`data/utils`), and `numpy/torch` interface (`_src/`) if, needed.
3. If it goes well start looking at `encoder-decoder` architectures (e.g. Brunton)

###### Pietro:
- [x] Add tests to cover most of the implemented code

- Make a simple example in which the pipeline is showcased
***
## Past roadmap
1. `koopman_estimators` contain everything is needed to fit the Koopman operator
    1. Implement the `sklearn` estimators as subclasses of the `BaseKoopmanEstimator` class using the functions in the folder `_algorithms`
    2. Testing
2. Folder `models` includes the subclasses of the `GeneralModel` defined in `main.py`.
    1. Kernel Koopman regression (no feature learning)
    2. DPNets
3. `DNNs` contains the deep learning architectures 
4. `feature_map` contains everything needed to define a feature map
5. `utils` folder: utilities + functions to interface with `torch`
 and `JAX`.
6. Visualization section
7. Adapt the quantiles + volatility functions on the covariance 

## On the structure of models:
The proposed `encoder->koop_estimation->decoder` paradigm does not fit well to every case. For example, the Kernel models are not expressed as such. In general, also the Primal models with fixed feature map.

Dependencies:
1. numpy
2. scipy
3. torch
4. torch-lightning
5. jax
6. pandas
