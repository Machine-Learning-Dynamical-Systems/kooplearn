---
title: 'kooplearn: A Scikit-Learn Compatible Library of Algorithms for Evolution Operator Learning'
tags:
  - Python
  - dynamical systems
  - evolution operator
  - koopman operator
  - transfer operator
  - operator learning
  - machine learning
  - representation learning
authors:
  - name: Giacomo Turri
    orcid: 0000-0002-3405-9292
    affiliation: 1
  - name: Grégoire Pacreau
    affiliation: 2
  - name: Giacomo Meanti
    orcid: 0000-0002-4633-2954
    affiliation: 3
  - name: Timothée Devergne
    orcid: 0000-0001-8369-237X
    affiliation: "4, 1"
  - name: Daniel Ordoñez
    orcid: 0000-0002-9793-2482
    affiliation: 1
  - name: Erfan Mirzaei
    orcid: 0000-0001-8720-1558
    affiliation: 1
  - name: Bruno Belucci
    affiliation: 5
  - name: Karim Lounici
    orcid: 0000-0001-6806-6303
    affiliation: 2
  - name: Vladimir Kostic
    orcid: 0000-0002-2876-8834
    affiliation: "1, 6"
  - name: Massimiliano Pontil
    orcid: 0000-0001-9415-098X
    affiliation: "1, 7"
  - name: Pietro Novelli
    orcid: 0000-0003-1623-5659
    affiliation: 1
affiliations:
 - name: CSML Unit, Italian Institute of Technology, Genoa, Italy
   index: 1
 - name: CMAP, École Polytechnique, Palaiseau, France
   index: 2
 - name: Centre Inria de l’Université Grenoble Alpes, Montbonnot, France
   index: 3
 - name: ATSIM Unit, Italian Institute of Technology, Genoa, Italy
   index: 4
 - name: Paris Dauphine University, Paris, France
   index: 5
 - name: Faculty of Science, University of Novi Sad, Serbia
   index: 6
 - name: Dept. of Computer Science, University College London, U.K.
   index: 7
date: 9 January 2026
bibliography: paper.bib
---

# Summary

`kooplearn` is a machine-learning library that implements linear, kernel, and deep-learning estimators of *dynamical operators* and their spectral decompositions. `kooplearn` can model both discrete-time evolution operators (Koopman/Transfer) and continuous-time infinitesimal generators. By learning these operators, users can analyze dynamical systems via spectral methods, derive data-driven reduced-order models, and forecast future states and observables. `kooplearn`'s interface is compliant with the `scikit-learn` API [@sklearn], facilitating its integration into existing machine learning and data science workflows. Additionally, `kooplearn` includes curated benchmark datasets to support experimentation, reproducibility, and the fair comparison of learning algorithms. The software is available at <https://github.com/Machine-Learning-Dynamical-Systems/kooplearn>.

# Statement of Need
From fluid flows down to atomistic motions, dynamical systems permeate every scientific discipline. Among the data-driven frameworks for modeling dynamical systems, evolution operator learning [@kostic2022] is both general and principled, and is especially well suited for interpretability [@schutte2001transfer; @mezic2005] and dimensionality reduction [@klus2018data]. An evolution operator $\mathsf{E}$ characterizes dynamical systems, either stochastic $x_{t + 1} \sim p(\cdot | x_{t})$, or deterministic $x_{t+1} \sim \delta(\cdot - F(x_{t}))$, as follows: for every function $f$ of the state of the system, $(\mathsf{E} f)(x_{t})$ is the expected value of $f$ one step ahead in the future, given that at time $t$ the system was found in $x_t$
$$(\mathsf{E} f)(x_t) = \int p(dy | x_{t}) f(y) = \mathbb{E}_{y \sim X_{t + 1} | X_{t}}[f(y) | x_t].$$
Notice that $\mathsf{E}$ is an operator because it maps any function $f$ to another function, $x_{t} \mapsto (\mathsf{E} f)(x_t)$, and is *linear* because $\mathsf{E}(f + \alpha g) = \mathsf{E} f + \alpha \mathsf{E} g$. When the dynamics is deterministic, $\mathsf{E}$ is known as the *Koopman operator* [@koopman1931], while in the stochastic case it is known as the *transfer operator* [@applebaum2009]. Arguably, the most important feature of evolution operators is their spectral decomposition [@mezic2005], which can be used to express the dynamics as a linear superposition of *modes*. These ideas lie at the core of the celebrated Time-lagged Independent Component Analysis [@Molgedey1994], and  Dynamical Mode Decomposition (DMD) [@schmid2010; @kutz2016]. 

Evolution operator learning is best understood from the perspective of *latent linear dynamical models*, which is schematically depicted in \autoref{fig:evop_scheme}. In this framework, the dynamical state $x_t$ is first mapped into a latent space defined by a (fixed or learned) representation $\varphi$. Then, a *linear evolution* map $E$ is learned to approximate the dynamics of the latents. The pair $(\varphi, E)$ provides an approximation of $\mathsf{E}$ restricted to the $d$-dimensional subspace spanned by the components of $\varphi$, given the data. `kooplearn` implements state-of-the-art methods to learn $\varphi$, $E$, and the associated spectral decomposition of $\mathsf{E}$. 

![Sketch of the action of an evolution operator on a protein folding trajectory. The dynamics of the protein is linearized by means of a nonlinear representation $\varphi$ and consequently evolved by means of the linear map $E$.\label{fig:evop_scheme}](Fig1.png){ width=100% }

The ecosystem of Python libraries that support operator-based modeling has grown considerably in recent years, with a predominant focus on the DMD family of methods. `PyDMD` [@pydmd] emphasizes classical and kernel DMD variants; `pykoopman` [@pykoopman] implements classical DMD methods with dictionary-based feature maps; `pykoop` [@pykoop] offers a modular framework for lifting-function construction with a focus on system identification and control; `DLKoopman` [@dlkoopman] focuses on autoencoder approaches, while `KoopmanLab` [@koopmanlab] targets Koopman neural operators. `kooplearn` addresses the general problem of learning evolution operators, and it is the result of a multi-year research effort in innovative operator learning algorithms. While it provides standard prediction and spectral decomposition utilities, it extends the state of the art in evolution operator learning codes by implementing fast kernel estimators [@meanti2023; @turri2025randomized], infinitesimal generator models for SDEs [@kostic2024generator; @devergne2024biased], and specialized losses for deep representation learning [@mardt2018; @kostic2024learning; @kostic2024neural; @turri2025self]. We now provide a concise overview of the functionality of `kooplearn`.

## Learning Linear Evolution Maps $E$
`kooplearn` implements state-of-the-art algorithms for learning evolution operators when the representation $\varphi$ is fixed. The library offers estimators in both their linear and kernel formulations (see the `Ridge` and `KernelRidge` classes), which bridge the gap between recent theoretical advances [@kostic2022; @kostic2023; @kostic2024consistent; @kostic2024learning] and practical code implementations. A key model in `kooplearn` is the kernel-based *Reduced Rank Regression* [@kostic2022]. This estimator provably outperforms traditional methods [@williams2015_kdmd] in approximating the operator's spectrum [@kostic2023], as illustrated in \autoref{fig:eigfns_approximation}. To our knowledge, `kooplearn` provides the only open-source implementation of this algorithm. To handle large datasets, `kooplearn` also includes randomized [@turri2025randomized] and Nyström-based [@meanti2023] kernel estimators, which significantly speed up the fitting process, making it one of the fastest libraries for kernel-based operator learning, as shown in \autoref{fig:fast_kernel}.

![Comparison between kernel DMD (kDMD) and Reduced Rank estimators. The Reduced Rank estimator provides a more accurate approximation of the leading eigenfunctions of the transfer operator for the overdamped Langevin dynamics.\label{fig:eigfns_approximation}](Fig2.png){ width=100% }

![Fit time of a Kernel model (Gaussian kernel) on a dataset of $5000$ observations from the Lorenz 63 dynamical system. The results are the median of three independent runs on a system equipped with an Intel Core i9-9900X CPU (3.50GHz) and 48GB of RAM memory.\label{fig:fast_kernel}](Fig3.png){ width=100% }

## Learning the Representation $\varphi$
`kooplearn` also exposes theoretically-grounded loss functions --- implemented in both PyTorch [@pytorch] and JAX [@jax] --- suited for learning the representation $\varphi$ with neural network models. This allows the incorporation of structural priors, such as graph-based encoders. Within this deep-learning approach, two main families are supported: (i) *encoder-decoder* schemes with the loss proposed in @lusch2018, and (ii) *encoder-only* schemes, for which `kooplearn` implements the VAMP loss [@mardt2018] and the spectral contrastive loss [@turri2025self].

## Learning the Infinitesimal Generator of Diffusion Processes
In continuous-time dynamics, the system's evolution operator can be expressed as the exponential of the *infinitesimal generator* $\mathsf{L}$, a differential operator defined by the equations of motion [@applebaum2009] (Chapter 3). Formally, for time-homogeneous dynamics, the generator relates to the evolution operator via $\mathsf{E} = e^{\mathsf{L}}$, and consequently $\mathbb{E}[f(X_t)\vert x_0] = (e^{t \mathsf{L}}f)(x_0)$. Since the exponential of an operator preserves its eigenfunctions, one can use knowledge of $\mathsf{L}$ (or its properties) to learn dynamical behavior without requiring lag-time data. In other words, it becomes possible to construct a physics-informed kinetic model $\mathsf{E}$ solely from static (equilibrium) data. To this end, `kooplearn` provides implementations of recent kernel-based algorithms for diffusion processes with Dirichlet boundary conditions from [@kostic2024generator], as well as neural representations as proposed in [@devergne2024biased]. As demonstrated in [@devergne2025slow], these approaches improve sample complexity compared to estimators that rely solely on lag-time trajectory data.

## Datasets
![Samples from the datasets included in `kooplearn`.\label{fig:datasets}](Fig4.png){ width=100% }

To foster reproducibility and rigorous benchmarking, `kooplearn` includes the `kooplearn.datasets` module, containing utilities to easily generate trajectories for systems that range from deterministic chaos (e.g., *Lorenz-63*, *Duffing oscillator*, *Logistic Map*) to stochastic and metastable dynamics (e.g., *stochastic linear systems*, *regime-switching models*, *Langevin dynamics*). A distinguishing feature of the library is the inclusion of benchmarks with accessible ground-truth spectral decompositions—such as the *Noisy Logistic Map* [@ostruszka2000dynamical] and *Overdamped Langevin Dynamics* in a quadruple-well potential [@Prinz2011]. These allow users to quantify the accuracy of learned eigenvalues and eigenfunctions directly (as demonstrated in \autoref{fig:eigfns_approximation}). Finally, the suite includes the *Ordered MNIST* from [@kostic2022] to evaluate performance on high-dimensional structured data. Examples of trajectories generated using the `kooplearn.datasets` module are illustrated in \autoref{fig:datasets}.

# Conclusion
`kooplearn` closely follows the `scikit-learn` API [@sklearn] and strives to lower the technical barrier to experimenting with evolution operators. At the same time, it provides optimized implementations of state-of-the-art algorithms for evolution operator learning, making it valuable for research, education, rapid prototyping, and exploratory analysis of dynamical systems. As of today, `kooplearn` has been employed in a variety of studies [@kostic2022; @bevanda2023; @kostic2023; @kostic2024consistent; @kostic2024learning; @turri2025self; @bevanda2025]. It can be installed using the command `pip install kooplearn`. Its documentation, alongside many worked-out examples, is available on the webpage <https://kooplearn.readthedocs.io/>.

# Acknowledgements

This work was partially funded by the European Union - NextGenerationEU and by the Ministry of University and Research (MUR), National Recovery and Resilience Plan (NRRP), through the PNRR MUR Project PE000013 CUP J53C22003010006 "Future Artificial Intelligence Research (FAIR)" and EU Project ELIAS under grant agreement No. 101120237.

# References