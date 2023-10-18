# Roadmap
#### Oct 17, 2023:
Added branch `in_out` in which I will work on serialization of each model, and more generally on every input-output utility which might be needed.

- The easiest storage option is just using `pickle`. `sklearn` as well as `lightning` models should be serializable by default.
- With [PEP 574](https://peps.python.org/pep-0574/), `pickle` introduced a 'protocol 5' to efficiently serialize large objects as Numpy Array.
- Implement a test for each serialization. 
- I should be able to save and load models and feature maps, as specified in `kooplearn.abc`.
- Add the possibility to pass both a path-like or a file-like object. The file-like object can be then used with `fsspec`.


#### Oct 3, 2023:
Macro implementations left to do:
- Nystrom Kernel methods (add a `NystromKernelDMD` model)
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
For the kernel DMD I have dropped the custom kernel objects defined in `kooplearn._src.kernels` and relied instead on `sklearn.gaussian_processes.kernels`, which better documented, supported and allow for easy HP tuning. Minimal API changing, working to have running tests. Check that `ExtendedDMD` models are working too.
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
