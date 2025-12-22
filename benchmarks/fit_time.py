#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "kooplearn",
#     "pydmd",
#     "pykoop",
#     "pykoopman",
#     "tyro",
#     "setuptools",
#     "matplotlib",
#     "derivative",
#     "lightning",
#     "tqdm"
# ]
# ///
import functools
import json
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import tyro
from tqdm import tqdm

# Ignore SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


@dataclass
class BenchmarkConfig:
    num_train_samples: int = 10000
    num_test_samples: int = 1000
    num_repeats: int = 3
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

    buffer = 1000
    if config.dataset == "lorenz":
        data = kooplearn.datasets.make_lorenz63(
            np.ones(3),
            n_steps=buffer
            + config.num_train_samples
            + buffer
            + config.num_test_samples,
        )
    elif config.dataset == "noisy_logistic":
        data = kooplearn.datasets.make_logistic_map(
            0.5,
            n_steps=buffer
            + config.num_train_samples
            + buffer
            + config.num_test_samples,
            M=10,
            random_state=config.random_seed,
        )
    elif config.dataset == "prinz_potential":
        gamma = 1.0
        sigma = 2.0
        data = kooplearn.datasets.make_prinz_potential(
            X0=0,
            n_steps=buffer
            + config.num_train_samples
            + buffer
            + config.num_test_samples,
            gamma=gamma,
            sigma=sigma,
            random_state=config.random_seed,
        )
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    dataset = {
        "train": data[buffer : buffer + config.num_train_samples].values,
        "test": data[-config.num_test_samples :].values,
    }
    return dataset


def kooplearn_PCR_runner(
    train_data: np.ndarray, test_data: np.ndarray, configs: BenchmarkConfig, random_state: int = 0
):
    from kooplearn.kernel import KernelRidge

    # Reduced Rank Regression
    model = KernelRidge(
        n_components=configs.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=configs.alpha,
        eigen_solver="arpack",
        random_state=random_state,
    )
    model = model.fit(train_data)

    # One-step prediction for RMSE
    X_test = test_data[:-1]
    Y_test = test_data[1:]
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    return rmse


def kooplearn_nystroem_PCR_runner(
    train_data: np.ndarray, test_data: np.ndarray, configs: BenchmarkConfig, random_state: int = 0
):
    from math import sqrt

    from kooplearn.kernel import NystroemKernelRidge

    n_centers = int(train_data.shape[0]*0.1)
    # Reduced Rank Regression
    model = NystroemKernelRidge(
        n_components=configs.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=configs.alpha,
        eigen_solver="arpack",
        n_centers=n_centers,  # 10% of data as centers
        random_state=random_state,
    )
    model = model.fit(train_data)

    # One-step prediction for RMSE
    X_test = test_data[:-1]
    Y_test = test_data[1:]
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    return rmse


def kooplearn_randomized_PCR_runner(
    train_data: np.ndarray, test_data: np.ndarray, configs: BenchmarkConfig, random_state: int = 0
):
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
        random_state=random_state,
    )
    model = model.fit(train_data)

    # One-step prediction for RMSE
    X_test = test_data[:-1]
    Y_test = test_data[1:]
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    return rmse


def pydmd_runner(
    train_traj: np.ndarray, test_traj: np.ndarray, configs: BenchmarkConfig, random_state: int = 0
):
    from pydmd.edmd import EDMDOperator

    np.random.seed(random_state)
    model = EDMDOperator(svd_rank=configs.rank, kernel_metric="rbf", kernel_params={})
    X = train_traj[:-1].T
    Y = train_traj[1:].T
    model = model.compute_operator(X, Y)

    # pydmd doesn't support predict, return NaN for RMSE
    rmse = np.nan

    return rmse


def pykoop_runner(
    train_traj: np.ndarray, test_traj: np.ndarray, configs: BenchmarkConfig, random_state: int = 0
):
    from math import sqrt

    import pykoop

    gamma = 1 / train_traj.shape[1]  # gamma = 2*self.shape
    n_components = int(train_traj.shape[0] * 0.1)  # 10% of data as features
    rfs = pykoop.RandomFourierKernelApprox(
        kernel_or_ft="gaussian",
        n_components=n_components,
        shape=gamma / 2.0,
        method="weight_offset",
        random_state=random_state,
    ).fit_transform(train_traj)
    model = pykoop.Edmd(alpha=configs.alpha).fit(rfs)

    # Create pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            (
                "rff",
                pykoop.KernelApproxLiftingFn(
                    kernel_approx=pykoop.RandomFourierKernelApprox(
                        kernel_or_ft="gaussian",
                        n_components=n_components,
                        shape=gamma / 2.0,
                        method="weight_offset",
                        random_state=random_state,
                    )
                ),
            )
        ],
        regressor=pykoop.Edmd(alpha=configs.alpha),
    )

    kp = kp.fit(train_traj)

    # One-step prediction for RMSE
    X_test = test_traj[:-1]
    Y_test = test_traj[1:]

    # Predict using the pipeline
    Y_pred = kp.predict_trajectory(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    return rmse


def pykoopman_runner(
    train_traj: np.ndarray, test_traj: np.ndarray, configs: BenchmarkConfig, random_state: int = 0
):
    import pykoopman as pk
    from pykoopman.regression import KDMD
    from sklearn.gaussian_process.kernels import RBF

    np.random.seed(random_state)
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
    model = model.fit(X, Y)

    # One-step prediction for RMSE
    X_test = test_traj[:-1]
    Y_test = test_traj[1:]
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    return rmse


runners_registry = {
    "kooplearn/PCR": kooplearn_PCR_runner,
    "kooplearn/nystroem_PCR": kooplearn_nystroem_PCR_runner,
    "kooplearn/randomized_PCR": kooplearn_randomized_PCR_runner,
    "pydmd/KDMD": pydmd_runner,
    "pykoopman/KDMD": pykoopman_runner,
    "pykoop/EDMD": pykoop_runner,
}


def run_benchmarks(configs: BenchmarkConfig) -> None:
    print(f"Starting benchmarks with configuration: {configs}")
    if configs.models == "all":
        runners = runners_registry.keys()
    elif isinstance(configs.models, str):
        runners = [configs.models]
    else:
        runners = configs.models

    print("Generating data...")
    dataset = make_data(configs)
    print("Data generation complete.")
    results = {}
    for runner_name in runners:
        print(f"Running benchmark for {runner_name}...")
        runner = runners_registry.get(runner_name)
        fit_times = []
        rmses = []
        if runner is None:
            raise ValueError(f"Unknown test {runner_name}")
        
        for rep in tqdm(range(configs.num_repeats), desc=runner_name, leave=False):
            try:
                rmse_single, fit_time_single = timer(runner)(
                    dataset["train"], dataset["test"], configs, random_state=rep
                )
                fit_times.append(fit_time_single)
                rmses.append(rmse_single)
            except Exception as e:
                fit_times.append(np.nan)
                rmses.append(np.nan)

        # Calculate median and IQR
        fit_time_median = np.nanmedian(fit_times)
        fit_time_q1 = np.nanpercentile(fit_times, 25)
        fit_time_q3 = np.nanpercentile(fit_times, 75)
        fit_time_iqr = fit_time_q3 - fit_time_q1
        
        rmse_median = np.nanmedian(rmses)
        rmse_q1 = np.nanpercentile(rmses, 25)
        rmse_q3 = np.nanpercentile(rmses, 75)
        rmse_iqr = rmse_q3 - rmse_q1

        results[runner_name] = {
            "fit_time_median": fit_time_median,
            "fit_time_iqr": fit_time_iqr,
            "fit_time_values": fit_times,
            "rmse_median": rmse_median,
            "rmse_iqr": rmse_iqr,
            "rmse_values": rmses,
        }

    print("All benchmarks complete.")

    if configs.save_json:
        print("Saving results to JSON...")
        # Convert NaN to null for JSON serialization
        json_results = {}
        for k, v in results.items():
            # Convert fit_time_values and rmse_values lists, replacing NaN with null
            fit_time_vals = [x if isinstance(x, (int, float)) and not np.isnan(x) else None for x in v["fit_time_values"]]
            rmse_vals = [x if isinstance(x, (int, float)) and not np.isnan(x) else None for x in v["rmse_values"]]
            
            json_results[k] = {
                "fit_time_median": v["fit_time_median"]
                if not isinstance(v["fit_time_median"], float) or not np.isnan(v["fit_time_median"])
                else None,
                "fit_time_iqr": v["fit_time_iqr"]
                if not isinstance(v["fit_time_iqr"], float) or not np.isnan(v["fit_time_iqr"])
                else None,
                "fit_time_values": fit_time_vals,
                "rmse_median": v["rmse_median"]
                if not isinstance(v["rmse_median"], float) or not np.isnan(v["rmse_median"])
                else None,
                "rmse_iqr": v["rmse_iqr"]
                if not isinstance(v["rmse_iqr"], float) or not np.isnan(v["rmse_iqr"])
                else None,
                "rmse_values": rmse_vals,
            }
        with open("fit_time_benchmarks.json", "w") as f:
            json.dump(json_results, f, indent=4)
    if configs.make_plots:
        print("Creating plots...")
        import matplotlib.pyplot as plt

        # Create fit_time plot
        processed_fit = []
        for k, v in results.items():
            # Format name: replace / and _ with space
            name = k.replace("/", " ").replace("_", " ")
            is_kooplearn = "kooplearn" in k

            # Check if value is float
            fit_val = v["fit_time_median"]
            fit_iqr = v["fit_time_iqr"]
            is_valid = isinstance(fit_val, (int, float)) and not np.isnan(fit_val)
            val = fit_val if is_valid else 0
            err = fit_iqr if is_valid and isinstance(fit_iqr, (int, float)) and not np.isnan(fit_iqr) else 0
            # Create raw_val for sorting: valid floats first, then failures (as inf)
            raw_val = fit_val if is_valid else float("inf")
            label = f"{fit_val:.2f}±{fit_iqr:.2f}s" if is_valid else "FAILED"

            processed_fit.append(
                {
                    "name": name,
                    "value": val,
                    "error": err,
                    "raw_val": raw_val,
                    "label": label,
                    "is_kooplearn": is_kooplearn,
                    "is_valid": is_valid,
                }
            )

        # Sort: valid floats ascending (fastest first), failures last
        processed_fit.sort(key=lambda x: x["raw_val"])

        names = [p["name"] for p in processed_fit]
        values = [p["value"] for p in processed_fit]
        errors = [p["error"] for p in processed_fit]

        fig, ax = plt.subplots(figsize=(10, 3))

        # Plot bars with error bars
        bars = ax.barh(names, values, xerr=errors, color="#5e3c99", height=0.6, capsize=5, error_kw={"elinewidth": 1.5})

        # Invert y-axis to have the first item (fastest) at the top
        ax.invert_yaxis()

        # Remove spines and x-axis
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.get_xaxis().set_visible(False)

        # Bold specific labels
        for tick_label, p in zip(ax.get_yticklabels(), processed_fit):
            tick_label.set_color("#555555")
            if p["is_kooplearn"]:
                tick_label.set_fontweight("bold")
                tick_label.set_color("black")

        # Add annotations
        max_val = max(values) if values else 1
        for bar, p in zip(bars, processed_fit):
            width = bar.get_width()
            err = p["error"]
            text_x = width + err + (max_val * 0.02)
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
        print("Plot saved to fit_time_benchmarks.png")

        # Create RMSE plot
        processed_rmse = []
        for k, v in results.items():
            # Format name: replace / and _ with space
            name = k.replace("/", " ").replace("_", " ")
            is_kooplearn = "kooplearn" in k

            # Check if value is float
            rmse_val = v["rmse_median"]
            rmse_iqr = v["rmse_iqr"]
            is_valid = isinstance(rmse_val, (int, float)) and not np.isnan(rmse_val)
            val = rmse_val if is_valid else 0
            err = rmse_iqr if is_valid and isinstance(rmse_iqr, (int, float)) and not np.isnan(rmse_iqr) else 0
            # Create raw_val for sorting: valid floats first, then failures (as inf)
            raw_val = rmse_val if is_valid else float("inf")
            label = f"{rmse_val:.4f}±{rmse_iqr:.4f}" if is_valid else "N/A"

            processed_rmse.append(
                {
                    "name": name,
                    "value": val,
                    "error": err,
                    "raw_val": raw_val,
                    "label": label,
                    "is_kooplearn": is_kooplearn,
                    "is_valid": is_valid,
                }
            )

        # Sort: valid floats ascending (lowest RMSE first), failures last
        processed_rmse.sort(key=lambda x: x["raw_val"])

        names = [p["name"] for p in processed_rmse]
        values = [p["value"] for p in processed_rmse]
        errors = [p["error"] for p in processed_rmse]

        fig, ax = plt.subplots(figsize=(10, 3))

        # Plot bars with error bars
        bars = ax.barh(names, values, xerr=errors, color="#c23e1d", height=0.6, capsize=5, error_kw={"elinewidth": 1.5})

        # Invert y-axis to have the first item (best) at the top
        ax.invert_yaxis()

        # Remove spines and x-axis
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.get_xaxis().set_visible(False)

        # Bold specific labels
        for tick_label, p in zip(ax.get_yticklabels(), processed_rmse):
            tick_label.set_color("#555555")
            if p["is_kooplearn"]:
                tick_label.set_fontweight("bold")
                tick_label.set_color("black")

        # Add annotations
        max_val = max(values) if values else 1
        for bar, p in zip(bars, processed_rmse):
            width = bar.get_width()
            err = p["error"]
            text_x = width + err + (max_val * 0.02)
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
        fig.savefig("rmse_benchmarks.png", dpi=300, bbox_inches="tight")
        print("Plot saved to rmse_benchmarks.png")

    for k, v in results.items():
        fit_time_median = v["fit_time_median"]
        fit_time_iqr = v["fit_time_iqr"]
        rmse_median = v["rmse_median"]
        rmse_iqr = v["rmse_iqr"]
        if isinstance(fit_time_median, float):
            fit_time_str = f"{fit_time_median:.6f}±{fit_time_iqr:.6f} s" if not np.isnan(fit_time_median) else "N/A"
        if isinstance(rmse_median, float):
            rmse_str = f"{rmse_median:.6f}±{rmse_iqr:.6f}" if not np.isnan(rmse_median) else "N/A"
        print(f"{k}: fit_time={fit_time_str}, rmse={rmse_str}")


if __name__ == "__main__":
    configs = tyro.cli(BenchmarkConfig)
    run_benchmarks(configs)
    print("Script finished.")
