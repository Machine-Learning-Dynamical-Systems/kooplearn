# Analysing molecular dynamics simulations

_Author:_ [Giacomo Meanti](https://github.com/Giodiro)

In this tutorial we will do a simple spectral analysis of a molecular dynamics simulation of the alanine dipeptide.
This small molecule is often used as a test bench for algorithms working on MD since it provides interesting, slow dynamics while being small and easy to work with.

We will start by training a Nystroem reduced rank regression model with the `kooplearn.models.NystroemKernel` class and a Gaussian kernel, and from there move to the spectral analysis of eigenvalues and eigenfunctions of the estimated Koopman operator.

Using the Nystroem approximation allows us to run a full analysis of MD data in a few seconds, without needing to subsample the trajectories too much (we will be training models on 25k and 50k points in a matter of seconds).

## Setup


```python
import importlib

for module in ['kooplearn', 'matplotlib']:
    try:
        importlib.import_module(module)
    except ImportError:
        %pip install {module}
```


```python
import os
from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.gaussian_process.kernels import RBF
from kooplearn.models import NystroemKernel
from kooplearn.data import traj_to_contexts
```

### Download the data

The data consists of 3 250ns long independent molecular dynamic runs of the alanine dipeptide.
The first file contains the backbone dihedral angles $\phi, \psi$. They are known to provide enough information to identify 
the slow transitions of the dipeptide.

We will not use the dihedrals for training but only for the subsequent analysis: 
we will show that a model trained on the heavy atom pairwise distances (contained in the second file we're downloading)
can recover the same slow transitions as can be done by analyzing the dihedrals.

In a more realistic example, one would not have access to the dihedrals but would still be interested in recovering
slow transitions of the molecule.
 
For more information about the data, see https://markovmodel.github.io/mdshare/ALA2/#alanine-dipeptide



```python
base_url = "http://ftp.imp.fu-berlin.de/pub/cmb-data/"
files = [
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz",
    "alanine-dipeptide-3x250ns-heavy-atom-distances.npz",
]
for file in files:
    if not os.path.isfile(file):
        urlretrieve(base_url + file, file)
```

We will use the first of three simulations for training and the second for testing.
Here we plot (part of) the test dataset, both distances and dihedrals. Notice how the dihedrals show a sort of switching behavior. 
This becomes more clear in the Ramachandran plot (plotting each configuration's $\phi$ vs. $\psi$ angle) where each cluster defines
a specific state in which the protein can be in.


```python
train_distances = np.load(files[1])["arr_0"]
test_distances = np.load(files[1])["arr_1"]
test_dihedrals = np.load(files[0])["arr_1"]
time_ns = np.arange(train_distances.shape[0], dtype=np.float_)*1e-3
```


```python
subsample = 20

x_ticks = np.arange(len(test_distances[::subsample]))[::2000]
x_tickslabels = [f"{x:.0f}" for x in (time_ns[::subsample])[::2000]]

fig, ax = plt.subplots(ncols=3, figsize=(15, 9))
ax[0].imshow(test_distances[::subsample, :].T, aspect=120, cmap='plasma', interpolation='nearest')
ax[0].set_ylabel("Atom distance")
ax[0].set_xlabel("Time [ns]")
ax[0].set_yticks([], [])
ax[0].set_xticks(x_ticks, x_tickslabels)
ax[1].imshow(test_dihedrals[::subsample, :].T, aspect=2750, cmap='plasma', interpolation='nearest')
ax[1].set_ylabel("Backbone dihedrals")
ax[1].set_xlabel("Time [ns]")
ax[1].set_yticks([], [])
ax[1].set_xticks(x_ticks, x_tickslabels)

ax[2].scatter(test_dihedrals[::20, 0], test_dihedrals[::20, 1], s=2, color = 'k', alpha = 0.2)
ax[2].set_xlabel("$\phi$")
ax[2].set_ylabel("$\psi$")
ax[2].set_xticks([-np.pi, 0, np.pi], ["$-\pi$", "$0$", "$\pi$"])
ax[2].set_yticks([-np.pi, 0, np.pi], ["$-\pi$", "$0$", "$\pi$"])
ax[2].set_title("Ramachandran plot")
ax[2].margins(0)
ax[2].set_aspect('equal')
```


    
![png](ala2_nys_tutorial_files/ala2_nys_tutorial_8_0.png)
    


## Model training

 - `tr_dist_ctx`: the training context (x and y). This uses atom distances
 - `ts_dist_ctx`: the test context (x and y) using atom distances
 - `ts_dih_ctx`: an auxiliary *test* context which contains dihedrals instead of atom distances


```python
def build_contexts(time_lag: int, subsample: int):
    tr_dist_ctx = traj_to_contexts(train_distances[::subsample], time_lag=time_lag)
    ts_dist_ctx = traj_to_contexts(test_distances[::subsample], time_lag=time_lag)
    ts_dih_ctx = traj_to_contexts(test_dihedrals[::subsample], time_lag=time_lag)
    return tr_dist_ctx, ts_dist_ctx, ts_dih_ctx
```


```python
time_lag = 10
subsample = 50

tr_dist_ctx, ts_dist_ctx, ts_dih_ctx = build_contexts(time_lag, subsample)

print(f"Training data size: {train_distances.shape}")
print(f"Training context with {subsample}x subsampling and time lag of {time_lag} units:")
print(f"\t{tr_dist_ctx.shape=}")
```

    Training data size: (250000, 45)
    Training context with 50x subsampling and time lag of 10 units:
    	tr_dist_ctx.shape=(4990, 2, 45)


Build the Nyström reduced rank regression (N-RRR) model used for training. Important hyperparameters are:
 - The kernel, here a Gaussian (RBF) kernel with length-scale 0.5. You can use all kernels supported by scipy here!
 - The regularizer `tikhonov_reg`, should be a small positive number to ensure a stable solution
 - The rank of the model to be trained: this depends on the system being analyzed.
 - The number of Nyström centers. Increasing this parameter makes the model slower to train but also more accurate.
 
Then we fit the estimator and make predictions on the test set.


```python
kernel = RBF(length_scale=0.5)
nys_rrr = NystroemKernel(
    kernel=kernel,
    reduced_rank=True,  # Set the estimator to RRR. If False the estimator is PCR.
    tikhonov_reg=1e-7,
    rank=5,
    num_centers=600, 
    rng_seed=0
)
```


```python
# Fit the Nystroem model
nys_rrr.fit(tr_dist_ctx)
# Predict on the test set and compute an error metric
X_pred = nys_rrr.predict(ts_dist_ctx)  # Here we must pass the `X` part of the context
X_true = ts_dist_ctx.lookforward(nys_rrr.lookback_len)  # This is the `Y` part of the test context
rmse_onestep = np.sqrt(np.mean((X_pred - X_true)**2))
print(f"Trained Nystroem RRR model with {nys_rrr.num_centers} centers. "
      f"Test RMSE on 1-step predictions = {rmse_onestep:.3f}")
```

    Trained Nystroem RRR model with 600 centers. Test RMSE on 1-step predictions = 0.031


## Spectral Analysis

Given the Koopman operator we've learned with Nystroem RRR we can analyze the dynamical system.

To do this we will use the **spectrum** of the operator: the eigenvalues and eigenvectors.

Each eigenpair describes a different component of the dynamics. Eigenvalues provide information about the *relaxation time-scales* of each component, larger eigenvalues correspond to slower processes, while smaller ones correspond to faster processes. On the other hand eigenvectors describe how each component varies in space. A common way to use the eigenvectors is to provide a dimensionality reduction of the original (high-dimensional) states onto the **slowest dynamics** corresponding to the eigenvectors with highest eigenvalues.

A special mention to the highest eigenpair which should have eigenvalue close to 1. This is the stationary dynamics, and is generally not used for spectral analysis.

First we use the `nys_rrr` object to compute the eigenvalues of the operator, and the eigenvectors.
Note that since the Koopman operator is technically an *infinite dimensional operator* it doesn't have eigenvectors but **eigenfunctions**. For this reason, in order to have a concrete, finite-dimensional representation, the eigenfunctions must be evaluated on some data-points. Here we evaluate the right eigenfunctions on the subsampled test distances. Here it doesn't really matter if you're using the `X` or `Y` part of the data.

Another thing to note is that the eigenvalues are returned in ascending order. We reverse the order to simplify the analysis later (since we're most interested in the highest eigenvalues!)


```python
from kooplearn.utils import topk
evals, evec_right = nys_rrr.eig(eval_right_on=ts_dist_ctx)
top_evals = topk(np.abs(evals), 4) # Take the Four largest eigenvalues in modulus
evals = evals[top_evals.indices]
evec_right = evec_right[:, top_evals.indices]
```

Now we compute the time-scales implied by the eigenvalues. We have to take into account the fact that we've subsampled the data, and that we have generated trajectories such that the Koopman operator predicts Y from X where the pairs are `time_lag` time-steps apart. Knowing that the trajectories have a time-step of 1ps, we can compute the implied time-scales!

Note that:
 - we exclude the top eigenvalue (which corresponds to the stationary distribution), since it theoretically has an infinite relaxation time-scale
 - the spectrum for this system decays very rapidly as there are only two or three slow components to the dynamics. Since the other components are much faster, their true eigenvalues are very close to 0. The learned estimator will have some small errors due to having access to a finite dataset, so these eigenvalues might be estimated to be slightly lower than zero which causes them to have an **undefined time-scale**.


```python
tscales = -1 / np.log(evals.real.clip(1e-8, 1))
tscales_real = tscales * time_lag * subsample
print(f"Stationary distribution eigenvalue: {evals[0].real:.3f}")
print(f"Other eigenvalues: {evals[1:].real}")
print(f"Implied time-scales: {tscales_real[1:]*1e-3} ns")
```

    Stationary distribution eigenvalue: 1.000
    Other eigenvalues: [0.62691671 0.03133949 0.03133949]
    Implied time-scales: [1.07079777 0.14438864 0.14438864] ns


Finally, we're going to plot the first three eigenfunctions (excluding the stationary distribution) on the test data. We will superimpose the eigenfunctions, which in the plot below are represented with as the color, to the Ramachandran plot.
Since we know that the Ramachandran plot provides a good clustering of the dynamics of the alanine dipeptide, we wish to see whether the color (eigenfunctions) also matches the clusters well.

By matching each eigenfunction to an eigenvalue we will also learn which transition between different areas of the Ramachandran plot is slowest.


```python
fig, axes = plt.subplots(ncols=3, figsize=(12, 5))
cmap = plt.cm.coolwarm
alpha = 0.5
s = 5
efun_vals = evec_right.real

# It might be useful to play with the mid value and range of the
# color-maps for each eigenfunction to get nicer visualizations.
vcenters = [0, 0, 0]  # np.median(efun_vals, axis=0)
halfranges = [0.01, 0.1, 0.02]  # efun_vals.std(axis=0)

for i in range(len(axes)):
    axes[i].scatter(
        ts_dih_ctx.data[:, 0, 0], ts_dih_ctx.data[:, 0, 1],
        c=efun_vals[:, i + 1], s=s, cmap=cmap, alpha=alpha,
        norm=colors.CenteredNorm(vcenter=vcenters[i], halfrange=halfranges[i])
    )
    axes[i].set_title(f"time-scale = {tscales_real[i + 1].real:.3f}ps")
    axes[i].set_xticks([-np.pi, 0, np.pi], ["$-\pi$", "$0$", "$\pi$"])
    axes[i].set_xlabel("$\phi$")
    axes[i].set_yticks([-np.pi, 0, np.pi])
    if i == 0:
        axes[i].set_yticklabels(["$-\pi$", "$0$", "$\pi$"])
        axes[i].set_ylabel("$\psi$")
    else:
        axes[i].set_yticklabels([])
    axes[i].set_aspect('equal')
    axes[i].margins(0)
```


    
![png](ala2_nys_tutorial_files/ala2_nys_tutorial_22_0.png)
    


A brief comment on the eigenfunction plot:
 - the first panel shows that the slowest transition is between the right and left parts of the Ramachandran plot (high to low $\phi$).
 - the second panel shows that the second slowest transition is between $\psi\approx 0$ and $\psi\approx \pi$.
 - the third panel seems to be a very noisy component which cannot be reliably interpreted.

## Consistency Analysis

An important aspect of a spectral analysis is that it should remain consistent across different time-lags.

We will analyze the implied time-scale (ITS) in terms of number of simulation steps (thus ignoring the real units of the simulation and the subsampling).

As we train models with increasing time lags (the time lag is the distance in number of simulation steps between x and y pairs), the estimated eigenvalues will become smaller since the model's time-unit is the single time-lag, and relaxation becomes faster in terms of # of time-lags. However the ITS should remain relatively stable.

As we increase the time lags even further, faster dynamics will have relaxation times which are faster than a single time-lag, hence the model won't be able to estimate those reliably.

We will see that the 1st ITS increases rapidly as we start increasing the time lags. This behavior is well documented in similar models. At some point however it converges, much before it becomes too fast to be resolved.

The 2nd ITS instead is much smaller, hence it won't be resolved with higher time-lags. However we find that it converges very quickly at small time-lags.

We first define a helper function to train a model and return its ITS at a specific time-lag


```python
def train_model(time_lag: int, subsample: int):
    tr_dist_ctx, ts_dist_ctx, _ = build_contexts(time_lag, subsample)
    kernel = RBF(length_scale=0.5)
    nys_rrr = NystroemKernel(
        kernel=kernel,
        reduced_rank=True,  # Set the estimator to RRR. If False the estimator is PCR.
        tikhonov_reg=1e-7,
        rank=5,
        num_centers=300, 
        rng_seed=0
    )
    
    nys_rrr.fit(tr_dist_ctx)
    
    X_pred = nys_rrr.predict(ts_dist_ctx)  # Here we must pass the `X` part of the context
    X_true = ts_dist_ctx.lookforward(nys_rrr.lookback_len)# This is the `Y` part of the test context
    rmse_onestep = np.sqrt(np.mean((X_pred - X_true)**2))

    evals = nys_rrr.eig()
    evals = evals[topk(np.abs(evals), 4).indices]

    tscales = -1 / np.log(evals.real.clip(1e-8, 1))
    tscales_real = tscales * time_lag
    
    return rmse_onestep, tscales_real
```

Now we train one model per time-lag. Note that since the Nystroem method is very efficient we can quickly train many models even with many time-points (subsample=5 means training with 50k points).


```python
error_d = {}
tscale_d = {}
for time_lag in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    error, tscale = train_model(time_lag, subsample=50)
    error_d[time_lag] = error
    tscale_d[time_lag] = tscale
```

Simple plot of the RMSE for different time-lags


```python
fig, ax = plt.subplots()
ax.plot(error_d.keys(), error_d.values(), lw=3, marker='o')
ax.set_xlabel("Time-lag [ps]")
ax.set_ylabel("RMSE")
ax.margins(x=0)
```


    
![png](ala2_nys_tutorial_files/ala2_nys_tutorial_30_0.png)
    



```python
fig, ax = plt.subplots(ncols=2, figsize=(9, 4.5))
time_lags = list(tscale_d.keys())
ax[0].plot(time_lags, [tscale[1] for tscale in tscale_d.values()], lw=3, marker='o')
ax[1].plot(time_lags, [tscale[2] for tscale in tscale_d.values()], lw=3, marker='o')
ax[0].plot([0, 100], [0, 100], lw=3, c='k')
ax[1].plot([0, 100], [0, 100], lw=3, c='k')
ax[0].set_xlabel("Lag time (steps)")
ax[0].set_ylabel("ITS (steps)")
ax[1].set_xlabel("Lag time (steps)")
ax[1].set_ylabel("ITS (steps)")
ax[0].set_title("1st component")
ax[1].set_title("2nd component")
ax[0].margins(x=0)
ax[1].margins(x=0)
```


    
![png](ala2_nys_tutorial_files/ala2_nys_tutorial_31_0.png)
    


Note that if we correct for subsampling the first ITS is approximately 1000ps and the second is approximately 90ps.
