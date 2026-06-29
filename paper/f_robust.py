import time
from pathlib import Path

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from b_Closed_form import BS_barrier, BS_vanilla
from c_MC import generate_BS_paths_gpu, generate_heston_paths_gpu
from d_FDM import (
    CN_BS_barrier,
    CN_BS_vanilla,
    CS_ADI_heston_barrier,
    CS_ADI_heston_vanilla,
)


ROBUSTNESS_VARIABLES_MAIN = {
    "bs": {
        "r": [0.0025, 0.0975],
        "sigma": [0.05, 0.80], # < 0.812
        "T": [0.15, 2.85], # σ < root(T)
    },
    "hes": {
        "r": [0.0025, 0.0975],
        "kappa": [1.3, 4.5],    # > 1.25
        "theta": [0.035, 0.15], # > 0.03125
        "xi": [0.15, 0.60],     # < 0.632
        "rho": [-0.95, -0.05],
        "Y0": [0.005, 0.15],
        "T": [0.15, 2.85],
    },
}

BS_ETA_INDEX = {
    "r": 2,
    "sigma": 3,
    "T": 4,
}

HESTON_ETA_INDEX = {
    "r": 2,
    "kappa": 3,
    "theta": 4,
    "xi": 5,
    "rho": 6,
    "Y0": 7,
    "T": 8,
}

OPTION_KEYS = ("van_call", "van_put", "barr_call", "barr_put")
OPTION_LABELS = {
    "van_call": "Vanilla Call",
    "van_put": "Vanilla Put",
    "barr_call": "Barrier Call",
    "barr_put": "Barrier Put",
}

def search_chunk_for_eta_indices(
    chunk_path,
    target_eta_idxs,
    row_batch_size=100_000,
    rdcc_nbytes=1024**2,
):
    chunk_path = Path(chunk_path)
    if not chunk_path.exists():
        return chunk_path.name, None

    target_eta_idxs = np.asarray(target_eta_idxs, dtype=np.int64)
    selected_batches = []

    with h5py.File(chunk_path, "r", rdcc_nbytes=rdcc_nbytes) as h5:
        data = h5["paths"]

        for start in range(0, data.shape[0], row_batch_size):
            end = min(start + row_batch_size, data.shape[0])
            eta_idx_batch = data[start:end, 0].astype(np.int64)
            local_positions = np.flatnonzero(
                np.isin(eta_idx_batch, target_eta_idxs)
            )
            if len(local_positions) == 0:
                continue

            global_positions = start + local_positions
            selected_batches.append(data[global_positions, :])

    if not selected_batches:
        return chunk_path.name, np.empty((0, 3), dtype=np.float64)
    return chunk_path.name, np.concatenate(selected_batches, axis=0)


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _synchronize_torch(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan, np.nan
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return mean, std


def _cvae_cases(
    cvae,
    checkpoint,
    model_name,
    eta,
    *,
    barrier,
    device,
    n_samples_list,
    repeats,
):
    eta_raw = np.asarray(eta[2:], dtype=np.float32)
    eta_min = _as_numpy(checkpoint["eta_min"]).astype(np.float32)
    eta_max = _as_numpy(checkpoint["eta_max"]).astype(np.float32)

    if eta_raw.shape != eta_min.shape or eta_raw.shape != eta_max.shape:
        raise ValueError(
            f"{model_name} eta shape={eta_raw.shape}, "
            f"checkpoint normalization shape={eta_min.shape}"
        )

    eta_scaled = (eta_raw - eta_min) / (eta_max - eta_min + 1e-8)
    eta_tensor = torch.as_tensor(eta_scaled, dtype=torch.float32, device=device)
    option_results = {
        key: {n: [] for n in n_samples_list}
        for key in OPTION_KEYS
    }
    seconds = {n: [] for n in n_samples_list}

    for n_samples in n_samples_list:
        for _ in range(repeats):
            _synchronize_torch(device)
            started = time.perf_counter()
            prices = cvae.total_pricing(
                eta_tensor,
                float(barrier),
                float(eta[1]),
                float(eta[2]),
                float(eta[-1]),
                int(n_samples),
            )
            _synchronize_torch(device)
            seconds[n_samples].append(time.perf_counter() - started)

            for key in OPTION_KEYS:
                option_results[key][n_samples].append(float(prices[key]))

    return {"prices": option_results, "seconds": seconds}


def _mc_total_pricing(
    model_name,
    eta,
    *,
    barrier,
    n_samples,
    dt,
    need_barrier,
):
    S0, K, r = float(eta[0]), float(eta[1]), float(eta[2])
    T = float(eta[-1])

    if model_name == "bs":
        if need_barrier:
            S_T, knocked = generate_BS_paths_gpu(
                eta, n_paths=n_samples, dt=dt, B=barrier
            )
        else:
            sigma = float(eta[3])
            z = cp.random.randn(n_samples)
            S_T = S0 * cp.exp(
                (r - 0.5 * sigma**2) * T + sigma * cp.sqrt(T) * z
            )
            knocked = cp.zeros(n_samples, dtype=bool)
    elif model_name == "hes":
        X_T, knocked = generate_heston_paths_gpu(
            eta,
            n_paths=n_samples,
            dt=dt,
            B=barrier if need_barrier else 0,
        )
        S_T = S0 * cp.exp(X_T)
    else:
        raise ValueError("model_name must be 'bs' or 'hes'.")

    call_payoff = cp.maximum(S_T - K, 0.0)
    put_payoff = cp.maximum(K - S_T, 0.0)
    alive = (~knocked).astype(call_payoff.dtype)
    discount = float(np.exp(-r * T))

    return {
        "van_call": discount * float(call_payoff.mean()),
        "van_put": discount * float(put_payoff.mean()),
        "barr_call": discount * float((call_payoff * alive).mean()),
        "barr_put": discount * float((put_payoff * alive).mean()),
    }

def _mc_cases(
    model_name,
    eta,
    *,
    barrier,
    barr_types,
    n_samples_list,
    repeats,
    dt,
):
    option_results = {
        key: {n: [] for n in n_samples_list}
        for key in OPTION_KEYS
    }
    seconds = {n: [] for n in n_samples_list}
    errors = []
    need_barrier = "barr" in barr_types

    for n_samples in n_samples_list:
        for repeat in range(repeats):
            cp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            try:
                prices = _mc_total_pricing(
                    model_name,
                    eta,
                    barrier=barrier,
                    n_samples=n_samples,
                    dt=dt,
                    need_barrier=need_barrier,
                )
                cp.cuda.get_current_stream().synchronize()
                seconds[n_samples].append(time.perf_counter() - started)
                for key in OPTION_KEYS:
                    option_results[key][n_samples].append(float(prices[key]))
            except Exception as exc:
                seconds[n_samples].append(time.perf_counter() - started)

                for key in OPTION_KEYS:
                    option_results[key][n_samples].append(np.nan)
                errors.append(
                    f"mc(n={n_samples}, repeat={repeat})="
                    f"{type(exc).__name__}: {exc}"
                )

    return {"prices": option_results, "seconds": seconds}, errors


def _pricing(price_fn):
    started = time.perf_counter()
    try:
        return float(price_fn()), time.perf_counter() - started, None
    except Exception as exc:
        return np.nan, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def _benchmark_case(model_name, eta, barr_type, option_type, *, barrier, run_fdm):
    row = {
        "closed": np.nan,
        "fdm": np.nan,
        "closed_seconds": np.nan,
        "fdm_seconds": np.nan,
    }
    errors = []

    if model_name == "bs":
        if barr_type == "van":
            closed_fn = lambda: BS_vanilla(eta, option_type)
        else:
            closed_fn = lambda: BS_barrier(eta, barrier, option_type)
        row["closed"], row["closed_seconds"], error = _pricing(closed_fn)
        if error:
            errors.append(f"closed={error}")

        if run_fdm:
            if barr_type == "van":
                fdm_fn = lambda: CN_BS_vanilla(eta, type=option_type)
            else:
                fdm_fn = lambda: CN_BS_barrier(eta, type=option_type, B=barrier)
            row["fdm"], row["fdm_seconds"], error = _pricing(fdm_fn)
            if error:
                errors.append(f"fdm={error}")

    elif model_name == "hes":
        if run_fdm:
            if barr_type == "van":
                fdm_fn = lambda: CS_ADI_heston_vanilla(eta, type=option_type, dv=0.0001)
            else:
                fdm_fn = lambda: CS_ADI_heston_barrier(
                    eta, type=option_type, B=barrier, dv=0.0001
                )
            row["fdm"], row["fdm_seconds"], error = _pricing(fdm_fn)
            if error:
                errors.append(f"fdm={error}")
    else:
        raise ValueError("model_name must be 'bs' or 'hes'.")

    return row, errors

def run_robustness(
    BS_eta,
    Hes_eta,
    robustness_variables=ROBUSTNESS_VARIABLES_MAIN,
    *,
    models=("bs", "hes"),
    barrier=0.8,
    barr_types=("van", "barr"),
    option_types=("call", "put"),
    run_fdm=True,
    run_mc=False,
    n_samples_list=(1_000, 10_000, 100_000),
    mc_repeats=50,
    mc_dt=0.001,
    cvae=None,
    checkpoint=None,
    device="cpu",
    cvae_repeats=50,
):
    models = tuple(models)
    barr_types = tuple(barr_types)
    option_types = tuple(option_types)
    n_samples_list = tuple(int(n) for n in n_samples_list)

    invalid_models = set(models) - {"bs", "hes"}
    if invalid_models:
        raise ValueError(f"Unknown models: {sorted(invalid_models)}")
    if set(barr_types) - {"van", "barr"}:
        raise ValueError("barr_types must contain only 'van' and/or 'barr'.")
    if set(option_types) - {"call", "put"}:
        raise ValueError("option_types must contain only 'call' and/or 'put'.")
    if not n_samples_list or any(n <= 0 for n in n_samples_list):
        raise ValueError("n_samples_list must contain positive integers.")
    if mc_repeats <= 0 or cvae_repeats <= 0:
        raise ValueError("repeat counts must be positive.")
    if cvae is not None and checkpoint is None:
        raise ValueError("checkpoint is required when cvae is provided.")
    if cvae is not None and len(models) != 1:
        raise ValueError("A loaded CVAE can be evaluated for exactly one model at a time.")

    specs = {
        "bs": (tuple(BS_eta), BS_ETA_INDEX),
        "hes": (tuple(Hes_eta), HESTON_ETA_INDEX),
    }
    rows = []
    case_number = 0
    largest_n = max(n_samples_list)

    for model_name in models:
        if model_name not in robustness_variables:
            raise KeyError(
                f"No robustness variables configured for model '{model_name}'."
            )
        base_eta, eta_index = specs[model_name]
        variable_values = robustness_variables[model_name]

        for variable, requested_values in variable_values.items():
            index = eta_index[variable]

            for value in requested_values:
                eta = list(base_eta)
                eta[index] = float(value)
                eta = tuple(eta)

                cvae_data = None
                cvae_error = None
                if cvae is not None:
                    try:
                        cvae_data = _cvae_cases(
                            cvae,
                            checkpoint,
                            model_name,
                            eta,
                            barrier=barrier,
                            device=device,
                            n_samples_list=n_samples_list,
                            repeats=cvae_repeats,
                        )
                    except Exception as exc:
                        cvae_error = f"cvae={type(exc).__name__}: {exc}"

                mc_data = None
                mc_errors = []
                if run_mc:
                    mc_data, mc_errors = _mc_cases(
                        model_name,
                        eta,
                        barrier=barrier,
                        barr_types=barr_types,
                        n_samples_list=n_samples_list,
                        repeats=mc_repeats,
                        dt=mc_dt,
                    )

                for barr_type in barr_types:
                    for option_type in option_types:
                        case_number += 1
                        print(
                            f"[{case_number:03d}] {model_name} | {variable}={value:g} "
                            f"| {barr_type} {option_type}"
                        )
                        benchmark, errors = _benchmark_case(
                            model_name,
                            eta,
                            barr_type,
                            option_type,
                            barrier=barrier,
                            run_fdm=run_fdm,
                        )
                        errors.extend(mc_errors)
                        if cvae_error:
                            errors.append(cvae_error)

                        option_key = f"{barr_type}_{option_type}"
                        mc_results = (
                            mc_data["prices"][option_key]
                            if mc_data is not None
                            else {n: [] for n in n_samples_list}
                        )
                        cvae_results = (
                            cvae_data["prices"][option_key]
                            if cvae_data is not None
                            else {n: [] for n in n_samples_list}
                        )
                        mc_mean, mc_std = _mean_std(mc_results[largest_n])
                        cvae_mean, cvae_std = _mean_std(cvae_results[largest_n])

                        rows.append({
                            "model": model_name,
                            "variable": variable,
                            "value": float(value),
                            "barr_type": barr_type,
                            "option_type": option_type,
                            "eta": eta,
                            **benchmark,
                            "mc": mc_mean,
                            "mc_std": mc_std,
                            "mc_seconds": (
                                float(np.nanmean(mc_data["seconds"][largest_n]))
                                if mc_data is not None else np.nan
                            ),
                            "mc_results": mc_results,
                            "mc_seconds_by_n": (
                                mc_data["seconds"] if mc_data is not None else {}
                            ),
                            "cvae": cvae_mean,
                            "cvae_std": cvae_std,
                            "cvae_seconds": (
                                float(np.nanmean(cvae_data["seconds"][largest_n]))
                                if cvae_data is not None else np.nan
                            ),
                            "cvae_results": cvae_results,
                            "cvae_seconds_by_n": (
                                cvae_data["seconds"] if cvae_data is not None else {}
                            ),
                            "error": "; ".join(dict.fromkeys(errors)) if errors else None,
                        })

    results = pd.DataFrame(rows)
    results.attrs["n_samples_list"] = n_samples_list
    results.attrs["mc_repeats"] = mc_repeats
    results.attrs["cvae_repeats"] = cvae_repeats
    results["reference"] = np.where(
        results["model"].eq("bs"), results["closed"], results["fdm"]
    )

    for method in ("fdm", "mc", "cvae"):
        difference = results[method] - results["reference"]
        results[f"{method}_abs_error"] = difference.abs()
        results[f"{method}_rel_error_pct"] = np.where(
            results["reference"].abs() > 1e-12,
            difference.abs() / results["reference"].abs() * 100.0,
            np.nan,
        )

    return results


def build_sampling_summary(results):
    n_samples_list = tuple(results.attrs.get("n_samples_list", ()))
    if not n_samples_list and not results.empty:
        n_samples_list = tuple(sorted(results.iloc[0]["cvae_results"].keys()))

    rows = []
    for _, result in results.iterrows():
        for n_samples in n_samples_list:
            mc_mean, mc_std = _mean_std(result["mc_results"].get(n_samples, []))
            cvae_mean, cvae_std = _mean_std(
                result["cvae_results"].get(n_samples, [])
            )
            reference = float(result["reference"])

            rows.append({
                "model": result["model"],
                "variable": result["variable"],
                "value": result["value"],
                "barr_type": result["barr_type"],
                "option_type": result["option_type"],
                "n_samples": n_samples,
                "eta": result["eta"],
                "closed": result["closed"],
                "fdm": result["fdm"],
                "reference": reference,
                "mc_mean": mc_mean,
                "mc_std": mc_std,
                "mc_rel_error_pct": (
                    abs(mc_mean - reference) / abs(reference) * 100.0
                    if np.isfinite(mc_mean) and abs(reference) > 1e-12 else np.nan
                ),
                "cvae_mean": cvae_mean,
                "cvae_std": cvae_std,
                "cvae_rel_error_pct": (
                    abs(cvae_mean - reference) / abs(reference) * 100.0
                    if np.isfinite(cvae_mean) and abs(reference) > 1e-12 else np.nan
                ),
                "mc_seconds_mean": _mean_std(
                    result["mc_seconds_by_n"].get(n_samples, [])
                )[0],
                "cvae_seconds_mean": _mean_std(
                    result["cvae_seconds_by_n"].get(n_samples, [])
                )[0],
                "error": result["error"],
            })

    return pd.DataFrame(rows)


def save_robustness_results(results, output_dir="result/robust", stem="robustness_results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / f"{stem}.pkl"
    csv_path = output_dir / f"{stem}_summary.csv"
    results.to_pickle(pickle_path)
    build_sampling_summary(results).to_csv(csv_path, index=False)
    return pickle_path, csv_path


def plot_sampling_comparison(results, model_name, variable, value):
    summary = build_sampling_summary(results)
    subset = summary[
        summary["model"].eq(model_name)
        & summary["variable"].eq(variable)
        & np.isclose(summary["value"], value)
    ]
    if subset.empty:
        raise ValueError(f"No result for {model_name} {variable}={value}.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    axes = axes.ravel()
    x_offset = 0.04

    for ax, option_key in zip(axes, OPTION_KEYS):
        barr_type, option_type = option_key.split("_", 1)
        option_data = subset[
            subset["barr_type"].eq(barr_type)
            & subset["option_type"].eq(option_type)
        ].sort_values("n_samples")
        n_samples = option_data["n_samples"].to_numpy(dtype=int)
        x_positions = np.arange(len(n_samples))

        if option_data["mc_mean"].notna().any():
            ax.errorbar(
                x_positions - x_offset,
                option_data["mc_mean"],
                yerr=1.96 * option_data["mc_std"],
                fmt="o-",
                color="grey",
                capsize=5,
                label="MC (95% range)",
            )
        if option_data["cvae_mean"].notna().any():
            ax.errorbar(
                x_positions + x_offset,
                option_data["cvae_mean"],
                yerr=1.96 * option_data["cvae_std"],
                fmt="o-",
                color="steelblue",
                capsize=5,
                label="CVAE (95% range)",
            )

        reference = float(option_data["reference"].iloc[0])
        reference_label = "Closed" if model_name == "bs" else "FDM"
        ax.axhline(
            reference,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=reference_label,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([
            f"{n // 1000}K" if n >= 1000 and n % 1000 == 0 else f"{n:,}"
            for n in n_samples
        ])
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_title(OPTION_LABELS[option_key])
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Option price")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"{model_name} robustness | {variable}={value:g}")
    plt.tight_layout()
    plt.show()
    return fig, axes