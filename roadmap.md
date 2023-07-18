# Roadmap
## Jul 18 '23
**[Bruno]**
1. Implement the [code for the paper](https://github.com/BethanyL/DeepKoopman) "Deep learning for universal linear embeddings of nonlinear dynamics" by Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton

**[Grégoire]**
1. Finish visualization tools for modes

**[Pie]**
1. Review Bruno's code for `dpnets`
    1. Add a `JAX` analogue model
    2. Test Bruno's implementation on an easy dataset
2. Everything from the [Jul 13](#jul-13-23) list

**[Vladi]**
1. Impact of the pre-processing/windowing steps on the estimators
2. Study how the spectral filtering regularization schemes interact with the estimation of the eigenvalues **especially** if they are complex.

### Features to add before private release:

- [ ] Visualization tools using `dash`
- [ ] I/O utilities
     - [ ] Time-series dataloader for `torch`, `jax`, and `numpy`
     - [ ] Context windows, chunking of the trajectory
     - [ ] Adding time-related features
- [ ] Metrics
    - [ ] Reconstruction losses
    - [ ] Probabilistic losses

- [ ] Confidence intervals with quantiles

## Jul 13 '23
**[Pietro]**
1. Add to the abstract `BaseModel` class two methods: `load` and `save` to save the estimators. Implement them for the `edmd` and `kernel` models.
2. Add a **robust** function that, given a list of complex numbers containing only pure reals and couples of complex conjugates, return two masks, one with the indices of the real numbers and one with the indices of only one of the elements of the CC pair.
   1. Add complementary function to perform checks and sum reductions
3. Modify the mode function to return **everything** (i.e. including the initial condition) except of the eigenvalue.

**[Vladi]**
1. Study how the spectral filtering regularization schemes interact with the estimation of the eigenvalues **especially** if they are complex. 

## Jul 3 '23
**[Grégoire]**: 
1. Add few synthetic financial datasets
2. Add a minimal implementation of the viz functionality (`matplotlib` for the moment. Look into `dask` and `gradio`)

**[Bruno]**
1. Implement a functioning `DPnets` feature map
2. Implement utility functions to: data loading (`data/utils`), and `numpy/torch` interface (`_src/`) if, needed.
3. If it goes well start looking at `encoder-decoder` architectures (e.g. Brunton)

**[Pietro]**
1. Add tests to cover most of the implemented code
2. Implement Nyström solvers
3. Make a simple example in which the pipeline is showcased
    

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
