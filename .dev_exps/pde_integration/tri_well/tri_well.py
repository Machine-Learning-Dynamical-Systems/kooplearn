import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
import ml_confs as mlcfg
from ml_confs import BaseConfigs
from pathlib import Path
from kme_utils import make_movie
from functools import partial
from kooplearn.models.kernel import KernelDMD
from kooplearn._src.kernels import RBF
from kooplearn.data.datasets import LangevinTripleWell1D as TriWell
import optuna

class NormalizedRBF(RBF):
    def __init__(self, length_scale: float):
        super().__init__(length_scale=length_scale)
    
    def __call__(self, X: ArrayLike, Y: Optional[ArrayLike] = None) -> ArrayLike:
        _out = super().__call__(X, Y)
        _norm = 2*np.pi*self.length_scale**2
        if np.ndim(X) == 1:
            _feat_dim = 1
        else:
            _feat_dim = X.shape[1]
        _out /= np.sqrt(_norm)**_feat_dim
        return _out

def generate_dataset(cfg: BaseConfigs, rng_seed: int = 42):
    ds = TriWell(
        gamma = cfg.gamma,
        kt = cfg.kt,
        dt = cfg.dt,
        rng_seed = rng_seed
    )
    num_init_pts = 200
    traj = ds.generate(np.linspace(-1, 1, num_init_pts), int(cfg.num_samples/num_init_pts), show_progress=True)
    traj = traj[::cfg.take_every]
    traj_in = traj[:-1].reshape(-1, 1)
    traj_out = traj[1:].reshape(-1, 1)
    return traj_in, traj_out, ds._ref_eigenvalues, ds._ref_boltzmann_density, ds._ref_domain_sample

def objective(trial: optuna.Trial, training_traj: ArrayLike, ref_eigenvalues: ArrayLike):
    length_scale = trial.suggest_float("length_scale", 0.001, 0.2)
    tikhonov_reg = trial.suggest_float("tikhonov_reg", 1e-7, 1e-3, log=True)
    kernel = NormalizedRBF(length_scale=length_scale)
    
    model = KernelDMD(
        kernel = kernel,
        rank = len(ref_eigenvalues),
        tikhonov_reg = tikhonov_reg,
        solver = 'arnoldi'
    )
    _ = model.fit(training_traj[:-1], training_traj[1:])
    eigenvalues = np.sort(model.eig())

    #Using the eigenvalue error as a metric. Usually not applicable, as true eigenvalues are unknown..
    return np.linalg.norm(eigenvalues - np.sort(ref_eigenvalues))

def main(cfg: BaseConfigs):
    traj_in, traj_out, ref_eigenvalues, ref_density, domain_sample = generate_dataset(cfg)
    print("Number of samples: {}".format(len(traj_in)))
    # objective_fn = partial(objective, training_traj=training_traj, ref_eigenvalues=ref_eigenvalues)
    # study_name = "tri_well"
    # storage_name = "sqlite:///{}.db".format(study_name)
    #Optuna study
    # study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    # study.optimize(objective_fn, n_trials=cfg.num_trials)

    # print("Number of finished trials: {}".format(len(study.trials)))
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: {}".format(trial.value))
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    
    #Refitting the best model
    # length_scale = trial.params["length_scale"]
    # tikhonov_reg = trial.params["tikhonov_reg"]

    length_scale = 0.05
    tikhonov_reg = 1e-5
    kernel = NormalizedRBF(length_scale=length_scale)
    
    model = KernelDMD(
        kernel = kernel,
        rank = len(ref_eigenvalues),
        tikhonov_reg = tikhonov_reg,
        solver = 'arnoldi'
    )
    
    model = model.fit(traj_in, traj_out)
    eigenvalues = np.sort(model.eig())
    print(eigenvalues)
    print(np.sort(ref_eigenvalues))
    #Make a movie
    make_movie(
        model,
        np.array([0.0]),
        ref_density,
        domain_sample,
        time_horizon=2.0,
        duration_in_seconds= 10.0,
        fps = 24
    )

if __name__ == "__main__":
    cfg = mlcfg.from_file(Path(__file__).parent / "tri_well_cfg.yaml")
    main(cfg)