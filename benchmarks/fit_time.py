#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "numpy==2.3.5",
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

import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm

import kooplearn.datasets
from kooplearn.kernel import KernelRidge, NystroemKernelRidge

# Ignore SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


@dataclass
class BenchmarkConfig:
    num_train_samples: int = 10000
    num_test_samples: int = 1000
    num_repeats: int = 10
    rank: int = 25
    alpha: float = 1e-6
    models: list[str] | str = "all"
    random_seed: int = 0
    save_json: bool = True
    make_plots: bool = False


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        return value, toc - tic

    return wrapper_timer


def make_data(config: BenchmarkConfig) -> dict[str, np.ndarray]:
    buffer = 1000
    total_steps = buffer + config.num_train_samples + buffer + config.num_test_samples

    data = kooplearn.datasets.make_lorenz63(np.ones(3), n_steps=total_steps)

    # Kooplearn datasets return a wrapper, .values extracts numpy array
    data_dict = {
        "train": data[buffer : buffer + config.num_train_samples].values,
        "test": data[-config.num_test_samples :].values,
    }
    return data_dict


# --- Runners ---


def kooplearn_PCR_runner(train, test, cfg, random_state=0):
    model = KernelRidge(
        n_components=cfg.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=cfg.alpha,
        eigen_solver="arpack",
        random_state=random_state,
    )
    model.fit(train)
    return _calc_rmse(model, test)


def kooplearn_nystroem_PCR_runner(train, test, cfg, random_state=0):
    n_centers = int(train.shape[0] * 0.1)
    model = NystroemKernelRidge(
        n_components=cfg.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=cfg.alpha,
        eigen_solver="arpack",
        n_centers=n_centers,
        random_state=random_state,
    )
    model.fit(train)
    return _calc_rmse(model, test)


def kooplearn_randomized_PCR_runner(train, test, cfg, random_state=0):
    model = KernelRidge(
        n_components=cfg.rank,
        reduced_rank=False,
        kernel="rbf",
        alpha=cfg.alpha,
        eigen_solver="randomized",
        iterated_power=1,
        n_oversamples=5,
        random_state=random_state,
    )
    model.fit(train)
    return _calc_rmse(model, test)


def pydmd_runner(train, test, cfg, random_state=0):
    from pydmd.edmd import EDMDOperator

    np.random.seed(random_state)

    model = EDMDOperator(svd_rank=cfg.rank, kernel_metric="rbf", kernel_params={})
    X, Y = train[:-1].T, train[1:].T
    model.compute_operator(X, Y)
    # pydmd doesn't support predict in this context straightforwardly
    return np.nan


def pykoop_runner(train, test, cfg, random_state=0):
    import pykoop

    gamma = 1 / train.shape[1]
    n_components = int(train.shape[0] * 0.1)

    kernel_approx = pykoop.RandomFourierKernelApprox(
        kernel_or_ft="gaussian",
        n_components=n_components,
        shape=gamma / 2.0,
        method="weight_offset",
        random_state=random_state,
    )

    kp = pykoop.KoopmanPipeline(
        lifting_functions=[
            ("rff", pykoop.KernelApproxLiftingFn(kernel_approx=kernel_approx))
        ],
        regressor=pykoop.Edmd(alpha=cfg.alpha),
    )
    kp.fit(train)

    X_test, Y_test = test[:-1], test[1:]
    Y_pred = kp.predict_trajectory(X_test)
    return np.sqrt(np.mean((Y_pred - Y_test) ** 2))


def pykoopman_runner(train, test, cfg, random_state=0):
    import pykoopman as pk
    from pykoopman.regression import KDMD
    from sklearn.gaussian_process.kernels import RBF

    np.random.seed(random_state)
    gamma = 1 / train.shape[1]
    length_scale = np.sqrt(0.5 / gamma)

    regressor = KDMD(
        svd_rank=cfg.rank,
        kernel=RBF(length_scale=length_scale),
        forward_backward=False,
        tikhonov_regularization=cfg.alpha,
    )
    model = pk.Koopman(regressor=regressor)
    model.fit(train[:-1], train[1:])
    return _calc_rmse(model, test)


def _calc_rmse(model, test_data):
    """Helper for standard fit/predict models."""
    X_test, Y_test = test_data[:-1], test_data[1:]
    Y_pred = model.predict(X_test)
    return np.sqrt(np.mean((Y_pred - Y_test) ** 2))


RUNNERS_REGISTRY = {
    "kooplearn/PCR": kooplearn_PCR_runner,
    "kooplearn/nystroem_PCR": kooplearn_nystroem_PCR_runner,
    "kooplearn/randomized_PCR": kooplearn_randomized_PCR_runner,
    "pydmd/KDMD": pydmd_runner,
    "pykoopman/KDMD": pykoopman_runner,
    "pykoop/EDMD": pykoop_runner,
}


# --- Plotting & Utils ---


def sanitize_for_json(obj):
    """Recursively replace NaNs with None for JSON standard compliance."""
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    return obj


def plot_benchmark(results: dict, metric: str, filename: str, color: str):
    """Generic plotting function for both time and RMSE."""
    processed = []

    metric_median = f"{metric}_median"
    metric_iqr = f"{metric}_iqr"

    for k, v in results.items():
        name = k.replace("/", " ").replace("_", " ")
        is_kooplearn = "kooplearn" in k

        val = v[metric_median]
        iqr = v[metric_iqr]

        is_valid = isinstance(val, (int, float)) and not np.isnan(val)

        # Sort key: Valid values first (asc), then failures
        sort_val = val if is_valid else float("inf")

        # Formatting
        if metric == "fit_time":
            label_text = f"{val:.2f}±{iqr:.2f}s" if is_valid else "FAILED"
        else:
            label_text = f"{val:.4f}±{iqr:.4f}" if is_valid else "N/A"

        processed.append(
            {
                "name": name,
                "value": val if is_valid else 0,
                "error": iqr if is_valid else 0,
                "sort_val": sort_val,
                "label": label_text,
                "is_kooplearn": is_kooplearn,
                "is_valid": is_valid,
            }
        )

    # Sort ascending
    processed.sort(key=lambda x: x["sort_val"])

    names = [p["name"] for p in processed]
    values = [p["value"] for p in processed]
    errors = [p["error"] for p in processed]

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.barh(
        names,
        values,
        xerr=errors,
        color=color,
        height=0.6,
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )

    ax.invert_yaxis()  # Fastest/Best on top
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.get_xaxis().set_visible(False)

    # Styling
    for tick_label, p in zip(ax.get_yticklabels(), processed):
        tick_label.set_color("#555555")
        if p["is_kooplearn"]:
            tick_label.set_fontweight("bold")
            tick_label.set_color("black")

    max_val = max(values) if values else 1
    for bar, p in zip(bars, processed):
        width = bar.get_width()
        text_x = width + p["error"] + (max_val * 0.02)
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

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {filename}")


def run_benchmarks(config: BenchmarkConfig) -> None:
    print(f"Starting benchmarks: {config}")

    runner_keys = (
        RUNNERS_REGISTRY.keys()
        if config.models == "all"
        else ([config.models] if isinstance(config.models, str) else config.models)
    )

    print("Generating data...")
    dataset = make_data(config)
    print("Data generation complete.")

    results = {}

    for name in runner_keys:
        runner = RUNNERS_REGISTRY.get(name)
        if not runner:
            raise ValueError(f"Unknown runner: {name}")

        fit_times, rmses = [], []

        for rep in tqdm(range(config.num_repeats), desc=name, leave=False):
            try:
                rmse, dt = timer(runner)(
                    dataset["train"], dataset["test"], config, random_state=rep
                )
                fit_times.append(dt)
                rmses.append(rmse)
            except Exception:
                fit_times.append(np.nan)
                rmses.append(np.nan)

        results[name] = {
            "fit_time_median": np.nanmedian(fit_times),
            "fit_time_iqr": np.nanpercentile(fit_times, 75)
            - np.nanpercentile(fit_times, 25),
            "fit_time_values": fit_times,
            "rmse_median": np.nanmedian(rmses),
            "rmse_iqr": np.nanpercentile(rmses, 75) - np.nanpercentile(rmses, 25),
            "rmse_values": rmses,
        }

    print("All benchmarks complete.")

    if config.save_json:
        print("Saving results to JSON...")
        clean_results = sanitize_for_json(results)
        with open("fit_time_benchmarks.json", "w") as f:
            json.dump(clean_results, f, indent=4)

    if config.make_plots:
        print("Creating plots...")
        plot_benchmark(results, "fit_time", "fit_time_benchmarks.svg", "#5e3c99")
        plot_benchmark(results, "rmse", "rmse_benchmarks.svg", "#c23e1d")

    # Summary print
    for k, v in results.items():
        ft, ft_iqr = v["fit_time_median"], v["fit_time_iqr"]
        rmse, rmse_iqr = v["rmse_median"], v["rmse_iqr"]

        ft_str = f"{ft:.4f}±{ft_iqr:.4f} s" if not np.isnan(ft) else "N/A"
        rmse_str = f"{rmse:.4f}±{rmse_iqr:.4f}" if not np.isnan(rmse) else "N/A"

        print(f"{k:<30} fit_time={ft_str:<20} rmse={rmse_str}")


if __name__ == "__main__":
    configs = tyro.cli(BenchmarkConfig)
    run_benchmarks(configs)
    print("Script finished.")
