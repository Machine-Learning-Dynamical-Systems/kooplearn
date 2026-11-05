# COMP0091: MSc Project Experiment Code
## Reference-free kernel selection for Koopman operator learning | September 2026

This sub-repository contains the **experiment and methodological code** for the MSc thesis _'There is a tide': Kernel effects on spectral learning of dynamical systems_, supervised by Prof. Massimiliano Pontil. 

This includes the pipelines that run the kernel spectral-learning experiments and compute the reference-free diagnostics, as well as the composite scoring. All of the files are designed to be part of an expansion of the MLDS `kooplearn` repository, and is actively being discussed with the authors of `kooplearn` at the time of writing. Therefore, the code in these files should not be considered in isolation, but as part of a larger library. Code used for plotting, table/CSV assembly, and the derived data are not included (please see Chapter 4 & Appendices of the thesis for the relevant results).

### Methodological contributions (`benchmarks/`)
This folder contains the python files holding the implemented diagnostics and scoring engine introduced in this project. These are the primary "concept $\rightarrow$ code" contributions by the author. 

Each file is an import-only module that is designed to work with, and calls on elements from, the pre-existing `kooplearn` library.

This folder is separated into the implementations of the numerous diagnostic axes ($x_1 to x_7$, see Table A.2 of the thesis), and the proposed scoring function:

- `diagnostics/` is structured to reflect the scoring hierarchy discussed in §5.1: per-mode level verification metrics live in `eigenpair.py`; candidate-representation level criteria are grouped in `representation.py`; the specific diagnostics they contain are detailed at the top of each file.
- `scoring/` consists of 
    -`spectral_analysis.py`, which aggregates spectral metrics from Kostic et al. 2023, and separately the long-horizon and encoder-inspired criteria which were adapted for spectral analysis;
    - `composite.py`, which is the central guarded composite scoring function; the weight sensitivity test referred to in §3.2.4 is also included here.

### Experimental pipelines (`experiments/`)
This folder consists of the self-contained notebooks which run the kernel-learning sweep for each kernel family (and encoders). 

Each notebook opens with a repeat of the metric pipeline and scoring function from `benchmarks/`, and any kernel helpers needed for the rest of the notebook. This is followed by the experiments, which is sectioned by dynamical system type (first the reversible baselines, then non-normal, then analytic controls), and has an experiment for each relevant system tested. For each DS, data is first generated from `kooplearn.datasets`, then calls functions from `kooplearn` for estimator fitting and spectral decomposition. Where the generator/system could not be found natively in `kooplearn`, it has been built from scratch inline. 

The spectral diagnostics and scoring found in `benchmarks/` are also repeated inline for each notebook, so that they can run against `kooplearn` alone. 

- The 4 notebooks experimenting on OU, Langevin, Duffing, Logistic map, Langevin + rotational skew, analytic linear system, and harmonic oscillator are 
    - `kernel_analysis_rbf.ipynb`, 
    - `kernel_analysis_poly.ipynb`, 
    - `kernel_analysis_linear.ipynb`,
    - `kernel_analysis_hermite.ipynb`.

    They cover the RBF, polynomial, linear, and Hermite kernel families. The first three are native kernel options in `kooplearn`; the Hermite kernel was built for this study and callable within the `kooplearn` setup, following the definition and parametric variations (goog/bad/ugly) given in Kostic et al. 2023.

- The graph random-walk systems demonstrate the over-smoothing failure domain in `kernel_analysis_graphs.ipynb`, with the three graph-based kernels implemented for this experiment: delta, diffusion, and resolvent-walk. More details can be found in §2.5.5.

- `kernel_analysis_ssl.ipynb` contains the learned-encoder self-supervised experiments with `SpectralContrastiveLearning` and the $P_*$-VAMP criterion, making clear how the project bridged encoder feature-learning to kernel representations.

- `joint_hyperparam_selection.ipynb`details the joint ($\gamma, \alpha$) tuning validation reported in §4.3.6.

**N.B. No free-standing plotting cells/code is included in this submission, since it does not bear on the results themselves and clutters up the notebooks. The decision was made to keep the notebooks and files streamlined and easier to assess by leaving them out.**

### Execution instructions
#### 1. Create an environment with Python 3.10+, and install the dependencies:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```

#### 2. For the per-family experiment pipelines,
either open interactively with

```sh
jupyter lab

```

then open a notebook and select `Run All`;
or run headless:

```sh
jupyter nbconvert --to notebook --execute experiments/kernel_analysis_[kernel family].ipynb

```

for each kernel family/pipeline option (`rbf`, `poly`, `hermite`, `linear`, `graphs`, `ssl`)


#### 3. The files in `benchmarks/` are *import-only* library modules,
not scripts. Individual functions, classes and variables can be imported from their `.py` file in Python, e.g.:

```python
# within the same relative path `0091/benchmarks/`
from diagnostics,eigenpair import (kernel_selection_criteria, composite_score, GRAND_WEIGHTS, HARD_CONSTRAINTS)

```