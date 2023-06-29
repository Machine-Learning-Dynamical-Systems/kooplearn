# Roadmap
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

Dependencies:
1. numpy
2. scipy
3. torch
4. torch-lightning
5. jax
6. pandas