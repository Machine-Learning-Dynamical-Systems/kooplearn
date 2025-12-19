#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kooplearn",
#     "pydmd",
#     "pykoop",
#     "pykoopman",
#     "tyro",
#     "setuptools",
#     "matplotlib"
# ]
# ///
import functools
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import tyro


@dataclass
class BenchmarkConfig:
    num_samples: int = 10000
    rank: int = 25
    alpha: float = 1e-6
    dataset: Literal["lorenz", "noisy_logistic", "prinz_potential"] = "lorenz"
    models: list[str] | str = "all"
    random_seed: int = 0
    save_json: bool = True
    make_plots: bool = False


def timer(func):
    # Adapted from https://realpython.com/python-timer/#creating-a-python-timer-decorator
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


def make_data(config: BenchmarkConfig) -> np.ndarray:
    import kooplearn.datasets

    if config.dataset == "lorenz":
        data = kooplearn.datasets.make_lorenz63(np.ones(3), n_steps=config.num_samples)
    elif config.dataset == "noisy_logistic":
        data = kooplearn.datasets.make_logistic_map(
            0.5, n_steps=config.num_samples, M=10, random_state=config.random_seed
        )
    elif config.dataset == "prinz_potential":
        gamma = 1.0
        sigma = 2.0
        data = kooplearn.datasets.make_prinz_potential(
            X0=0,
            n_steps=configs.num_samples,
            gamma=gamma,
            sigma=sigma,
            random_state=config.random_seed,
        )
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")
    return data.values


def kooplearn_PCR_runner(train_data: np.ndarray, configs: BenchmarkConfig):
    from kooplearn.kernel import KernelRidge

    # Reduced Rank Regression
    model = KernelRidge(
        n_components=configs.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=configs.alpha,
        eigen_solver="arpack",
        random_state=0,
    )
    _ = model.fit(train_data)


def kooplearn_nystroem_PCR_runner(train_data: np.ndarray, configs: BenchmarkConfig):
    from kooplearn.kernel import NystroemKernelRidge

    # Reduced Rank Regression
    model = NystroemKernelRidge(
        n_components=configs.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=configs.alpha,
        eigen_solver="arpack",
        n_centers=int(train_data.shape[0] * 0.05),  # 5% of data as centers
        random_state=0,
    )
    _ = model.fit(train_data)


def kooplearn_randomized_PCR_runner(train_data: np.ndarray, configs: BenchmarkConfig):
    from kooplearn.kernel import KernelRidge

    # Reduced Rank Regression
    model = KernelRidge(
        n_components=configs.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=configs.alpha,
        eigen_solver="randomized",
        iterated_power=1,
        n_oversamples=5,
        random_state=0,
    )
    _ = model.fit(train_data)


def pydmd_runner(train_traj: np.ndarray, configs: BenchmarkConfig):
    from pydmd.edmd import EDMDOperator

    model = EDMDOperator(svd_rank=configs.rank, kernel_metric="rbf", kernel_params={})
    X = train_traj[:-1].T
    Y = train_traj[1:].T
    _ = model.compute_operator(X, Y)


def pykoop_runner(train_traj: np.ndarray, configs: BenchmarkConfig):
    pass


def pykoopman_runner(train_traj: np.ndarray, configs: BenchmarkConfig):
    import pykoopman as pk
    from pykoopman.regression import KDMD
    from sklearn.gaussian_process.kernels import RBF

    gamma = 1 / train_traj.shape[1]
    length_scale = np.sqrt(0.5 / gamma)
    regressor = KDMD(
        svd_rank=configs.rank,
        kernel=RBF(length_scale=length_scale),
        forward_backward=False,
        tikhonov_regularization=configs.alpha,
    )

    model = pk.Koopman(regressor=regressor)
    X = train_traj[:-1]
    Y = train_traj[1:]
    model.fit(X, Y)


runners_registry = {
    "kooplearn/PCR": kooplearn_PCR_runner,
    "kooplearn/nystroem_PCR": kooplearn_nystroem_PCR_runner,
    "kooplearn/randomized_PCR": kooplearn_randomized_PCR_runner,
    "pydmd/PCR": pydmd_runner,
    "pykoopman/PCR": pykoopman_runner,
}


def run_benchmarks(configs: BenchmarkConfig) -> None:
    if configs.models == "all":
        runners = runners_registry.keys()
    elif isinstance(configs.models, str):
        runners = [configs.models]
    else:
        runners = configs.models

    data = make_data(configs)
    results = {}
    for runner_name in runners:
        runner = runners_registry.get(runner_name)
        if runner is None:
            raise ValueError(f"Unknown test {runner_name}")
        try:
            _, elapsed_time = timer(runner)(data, configs)
        except Exception as e:
            elapsed_time = repr(e)

        results[runner_name] = elapsed_time
    if configs.save_json:
        with open("fit_time_benchmarks.json", "w") as f:
            json.dump(results, f, indent=4)
    if configs.make_plots:
        import matplotlib.pyplot as plt

        # Preprocess data
        processed = []
        for k, v in results.items():
            # Format name: replace / and _ with space
            name = k.replace("/", " ").replace("_", " ")
            is_kooplearn = "kooplearn" in k

            # Check if value is float
            is_valid = isinstance(v, (int, float))
            val = v if is_valid else 0
            # Create raw_val for sorting: valid floats first, then failures (as inf)
            raw_val = v if is_valid else float("inf")
            label = f"{v:.2f}s" if is_valid else "FAILED"

            processed.append(
                {
                    "name": name,
                    "value": val,
                    "raw_val": raw_val,
                    "label": label,
                    "is_kooplearn": is_kooplearn,
                    "is_valid": is_valid,
                }
            )

        # Sort: valid floats ascending (fastest first), failures last
        processed.sort(key=lambda x: x["raw_val"])

        names = [p["name"] for p in processed]
        values = [p["value"] for p in processed]

        fig, ax = plt.subplots(figsize=(10, 3))

        # Plot bars
        bars = ax.barh(names, values, color="#5e3c99", height=0.6)

        # Invert y-axis to have the first item (fastest) at the top
        ax.invert_yaxis()

        # Remove spines and x-axis
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.get_xaxis().set_visible(False)

        # Bold specific labels
        for tick_label, p in zip(ax.get_yticklabels(), processed):
            tick_label.set_color("#555555")
            if p["is_kooplearn"]:
                tick_label.set_fontweight("bold")
                tick_label.set_color("black")

        # Add annotations
        max_val = max(values) if values else 1
        for bar, p in zip(bars, processed):
            width = bar.get_width()
            text_x = width + (max_val * 0.02)
            text_color = "#999999" if p["is_valid"] else "red"
            weight = "bold" if not p["is_valid"] else "normal"

            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                p["label"],
                va="center",
                color=text_color,
                fontweight=weight,
            )
        fig.savefig("fit_time_benchmarks.png", dpi=300, bbox_inches="tight")
    for k, v in results.items():
        if isinstance(v, float):
            v = f"{v:.6f} s"
        print(f"{k}: {v}")


if __name__ == "__main__":
    configs = tyro.cli(BenchmarkConfig)
    run_benchmarks(configs)
