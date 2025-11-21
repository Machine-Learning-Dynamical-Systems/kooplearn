import pickle

import numpy as np


def analyze_array_layout(arr, name):
    if not isinstance(arr, np.ndarray):
        return None
    return {
        "dtype": str(arr.dtype),
        "strides": str(arr.strides),
        "C_contig": arr.flags["C_CONTIGUOUS"],
        "F_contig": arr.flags["F_CONTIGUOUS"],
        "mean": arr.mean() if arr.size > 0 else 0.0,
    }


def extract_state_metadata(estimator):
    metadata = {}
    # Check top-level attributes
    for attr, value in vars(estimator).items():
        if isinstance(value, np.ndarray):
            metadata[attr] = analyze_array_layout(value, attr)
        elif isinstance(value, dict):
            # Check inside dicts (like _fit_result)
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    metadata[f"{attr}[{k}]"] = analyze_array_layout(v, f"{attr}[{k}]")
    return metadata


def debug_pickle_integrity(estimator_class, X, y=None):
    print(f"\n=== Debugging Pickle Integrity: {estimator_class.__name__} ===")

    # 1. Fit Original
    est = estimator_class()
    est.fit(X, y)
    orig_meta = extract_state_metadata(est)

    # 2. Pickle Roundtrip
    dump = pickle.dumps(est)
    est_loaded = pickle.loads(dump)
    load_meta = extract_state_metadata(est_loaded)

    # 3. Compare and Report
    fmt = "{:<25} {:<12} {:<30} {:<30} {:<10}"
    print("-" * 110)
    print(fmt.format("Attribute", "Property", "Original", "Loaded", "Status"))
    print("-" * 110)

    all_keys = sorted(set(orig_meta.keys()) | set(load_meta.keys()))

    for key in all_keys:
        orig = orig_meta.get(key)
        load = load_meta.get(key)

        if not orig or not load:
            print(
                fmt.format(
                    key, "EXISTENCE", str(bool(orig)), str(bool(load)), "MISSING"
                )
            )
            continue

        # Check Layout & Dtype
        props = ["dtype", "strides", "C_contig", "F_contig"]
        for p in props:
            if orig[p] != load[p]:
                print(fmt.format(key, p, str(orig[p]), str(load[p]), "FAIL"))

        # Check Value Drift
        if not np.isclose(orig["mean"], load["mean"], rtol=1e-15, atol=1e-15):
            print(
                fmt.format(
                    key,
                    "MEAN_VAL",
                    f"{orig['mean']:.18f}",
                    f"{load['mean']:.18f}",
                    "DRIFT",
                )
            )

    print("-" * 110)

    # 4. Prediction Check
    try:
        p1 = est.predict(X)
        p2 = est_loaded.predict(X)
        diff = np.abs(p1 - p2).max()
        print(f"Prediction Max Diff: {diff:.2e}")
        if diff > 1e-9:
            print("!!! SIGNIFICANT DRIFT DETECTED !!!")
    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    # Usage Example
    from sklearn.datasets import make_blobs

    from kooplearn.kernel import KernelRidge

    X, y = make_blobs(n_samples=50, n_features=5, random_state=42)
    debug_pickle_integrity(KernelRidge, X, y)
