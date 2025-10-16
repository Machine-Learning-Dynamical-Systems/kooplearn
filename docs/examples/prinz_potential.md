# Estimating Eigenfunctions of the Trasnfer Operator

_Author_: [Pietro Novelli](https://pietronvll.github.io/)


```python
from kooplearn.datasets import make_prinz_potential, compute_prinz_potential_eig
import matplotlib.pyplot as plt
from kooplearn.kernel import KernelRidge
from tqdm import tqdm
import numpy as np
from scipy.integrate import romb
from math import sqrt
```


```python
gamma = 1.0
sigma = 2.0
data = make_prinz_potential(X0 = 0, n_steps=int(1e6), gamma=gamma, sigma=sigma)
```


```python
def prinz_potential(x):
    return 4*(x**8+ 0.8*np.exp(-80*(x**2)) +  0.2*np.exp(-80*((x - 0.5)**2)) + 0.5*np.exp(-40*((x + 0.5)**2)))

def compute_boltzmann_density(gamma, sigma, xv):
    kT = sigma*sigma/(2*gamma)
    pdf = np.exp(-prinz_potential(xv)/kT)
    total_mass = romb(pdf, dx=xv[1] - xv[0])
    return pdf/total_mass
```


```python
xv = np.linspace(-3, 3, 2048 + 1)
density = compute_boltzmann_density(gamma, sigma, xv)
```


```python
plt.hist(data, bins=200, density=True, alpha=0.5, color="tab:purple");
plt.plot(xv, density, color='tab:purple')
plt.xlim(-1, 1)
```




    (-1.0, 1.0)




    
![png](prinz_potential_files/prinz_potential_5_1.png)
    



```python
data.reset_index().plot(x='time', y='x')
```




    <Axes: xlabel='time'>




    
![png](prinz_potential_files/prinz_potential_6_1.png)
    



```python
def estimate_eigfuns(reduced_rank: bool, n_repetitions: int = 1):
    xv = np.linspace(-2, 2, 2048 + 1)
    gamma = 1.0
    sigma = 2.0
    dt = 1e-4
    density = compute_boltzmann_density(gamma, sigma, xv)
    _, reference_eigfuns = compute_prinz_potential_eig(gamma,sigma, dt, eval_right_on = xv, num_components=5)
    reference_eigfuns = normalize_eigenfunctions(reference_eigfuns, xv, density)
    _, functions = map(
        np.array,
        zip(
            *(
                fit_and_estimate(reduced_rank, xv, density)
                for _ in tqdm(range(n_repetitions))
            )
        ),
    )
    for i in range(n_repetitions):
        for j in range(functions.shape[-1]):
            functions[i, :, j] = standardize_sign(functions[i, : ,j], reference_eigfuns[: ,j])
    return functions, reference_eigfuns, xv


def fit_and_estimate(reduced_rank, xv, density):
    subsample = 100
    gamma = 1.0
    sigma = 2.0
    data = make_prinz_potential(X0=0, n_steps=int(5e5), gamma=gamma, sigma=sigma).iloc[
        ::subsample
    ]
    model = KernelRidge(
        n_components=5, reduced_rank=reduced_rank, gamma=12.5, kernel="rbf", alpha=1e-6
    )
    model.fit(data)
    values, functions = model.eig(eval_right_on=xv[:, None])
    # Descending order
    values = np.flip(values)
    functions = np.flip(functions, axis=1)
    functions = normalize_eigenfunctions(functions, xv, density)
    return values, functions


def normalize_eigenfunctions(functions, xv, density):
    dx = xv[1] - xv[0]
    abs2_eigfun = (np.abs(functions) ** 2).T
    if density is not None:
        abs2_eigfun *= density
    # Norms
    funcs_norm = np.sqrt(romb(abs2_eigfun, dx=dx, axis=1))
    functions *= funcs_norm**-1.0
    return functions

def standardize_sign(current, reference):
    norm_p = np.linalg.norm(current + reference)
    norm_m = np.linalg.norm(current - reference)
    if norm_p <= norm_m:
        return -1.0*current
    else:
        return current
```


```python
n_repetitions = 10
results = {}
for method, reduced_rank in zip(["Principal Components (kDMD)", "Reduced Rank"], [False, True]):
    functions, reference_eigfuns, xv = estimate_eigfuns(reduced_rank, n_repetitions=n_repetitions)
    results[method] = functions
```

    100%|██████████| 10/10 [00:18<00:00,  1.89s/it]
    100%|██████████| 10/10 [01:02<00:00,  6.26s/it]



```python
fig, axs = plt.subplots(ncols=4, figsize=(9, 2))
for fun_id, ax in enumerate(axs):
    for method, functions in results.items():
        color = 'tab:blue' if 'Principal' in method else 'tab:orange' 
        label = method if fun_id == axs.shape[0] - 1 else None
        m = functions.mean(0)[:, fun_id]
        st = functions.std(0)[:, fun_id]
        ax.plot(xv, m, color=color, label=label)
        ax.fill_between(xv, m - st, m + st, color=color, alpha=0.3)
    ax.plot(xv, reference_eigfuns[:, fun_id], color='k', lw=1, ls='--', label='Reference' if fun_id == axs.shape[0] - 1 else None)
    ax.set_title(f"Eigenfunction $\psi_{fun_id}$")
    fig.legend()
    ax.set_xlim(-1, 1)
    #ax.set_ylim(-1.5, 1.5)
```

    /var/folders/hv/61b9_7692vs_qnbhp7fxkmfh0000gn/T/ipykernel_99097/3538860058.py:12: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
      fig.legend()



    
![png](prinz_potential_files/prinz_potential_9_1.png)
    

