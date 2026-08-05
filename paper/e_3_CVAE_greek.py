from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


OPTION_KEYS = ("van_call", "van_put", "barr_call", "barr_put")


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _model_device_dtype(cvae):
    parameter = next(cvae.parameters())
    return parameter.device, parameter.dtype


def _normalize_eta(
    eta_raw,
    eta_min,
    eta_max,
    check_bounds,
    parameter_name,
):
    eta_raw = np.asarray(eta_raw, dtype=np.float64)
    eta_min = np.asarray(eta_min, dtype=np.float64)
    eta_max = np.asarray(eta_max, dtype=np.float64)

    if eta_raw.shape != eta_min.shape or eta_raw.shape != eta_max.shape:
        raise ValueError(
            f"eta shape mismatch: eta={eta_raw.shape}, "
            f"eta_min={eta_min.shape}, eta_max={eta_max.shape}"
        )

    if check_bounds:
        below = eta_raw < eta_min
        above = eta_raw > eta_max
        if np.any(below | above):
            invalid = np.flatnonzero(below | above).tolist()
            raise ValueError(
                f"The '{parameter_name}' bump leaves the CVAE training range. "
                f"Out-of-range conditional indices: {invalid}. "
                "Use a smaller h or evaluate an interior parameter point."
            )

    return (eta_raw - eta_min) / (eta_max - eta_min + 1e-8)


def _make_common_noise(
    cvae,
    n_samples,
    antithetic,
    seed,
):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if antithetic and n_samples % 2 != 0:
        raise ValueError(
            "n_samples must be even when antithetic=True."
        )

    device, dtype = _model_device_dtype(cvae)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))

    if antithetic:
        half = n_samples // 2
        eps_z_half = torch.randn(
            (half, cvae.dim_z),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        eps_x_half = torch.randn(
            (half, cvae.dim_x),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        eps_z = torch.cat((eps_z_half, -eps_z_half), dim=0)
        eps_x = torch.cat((eps_x_half, -eps_x_half), dim=0)
    else:
        eps_z = torch.randn(
            (n_samples, cvae.dim_z),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        eps_x = torch.randn(
            (n_samples, cvae.dim_x),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    return eps_z, eps_x


@torch.no_grad()
def sample_cvae_with_noise(
    cvae,
    eta_scaled,
    eps_z,
    eps_x,
    batch_size = 65536,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if eps_z.ndim != 2 or eps_x.ndim != 2:
        raise ValueError("eps_z and eps_x must be two-dimensional tensors.")
    if eps_z.shape[0] != eps_x.shape[0]:
        raise ValueError("eps_z and eps_x must contain the same number of samples.")
    if eps_z.shape[1] != cvae.dim_z:
        raise ValueError(
            f"eps_z second dimension must be dim_z={cvae.dim_z}."
        )
    if eps_x.shape[1] != cvae.dim_x:
        raise ValueError(
            f"eps_x second dimension must be dim_x={cvae.dim_x}."
        )

    cvae.eval()
    device, dtype = _model_device_dtype(cvae)

    eta_scaled = torch.as_tensor(
        eta_scaled,
        dtype=dtype,
        device=device,
    )
    if eta_scaled.ndim != 1 or eta_scaled.numel() != cvae.dim_eta:
        raise ValueError(
            f"eta_scaled must have shape ({cvae.dim_eta},), "
            f"got {tuple(eta_scaled.shape)}."
        )

    outputs = []
    n_samples = eps_z.shape[0]

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        n_batch = end - start

        eta_batch = eta_scaled.unsqueeze(0).expand(n_batch, -1)

        mu_p, lv_p = cvae.prior(eta_batch)
        z = cvae.reparameterize(
            mu_p,
            lv_p,
            eps_z[start:end],
        )

        mu_x, lv_x = cvae.decoder(z, eta_batch)
        x = cvae.reparameterize(
            mu_x,
            lv_x,
            eps_x[start:end],
        )

        outputs.append(x.cpu())

    return torch.cat(outputs, dim=0)


def prices_from_cvae_samples(
    samples,
    S0,
    K,
    B,
    r,
    T,
):
    if samples.ndim != 2 or samples.shape[1] < 1:
        raise ValueError("samples must have shape (N, dim_x) with X_T in column 0.")
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K, and T must be positive.")
    if B is not None and not (0 < B < S0):
        raise ValueError("A down barrier must satisfy 0 < B < S0.")

    X_T = samples[:, 0] # samples[:, 0] = X_T and samples[:, 1] = M_T.
    S_T = float(S0) * torch.exp(X_T)

    call_payoff = torch.clamp(S_T - float(K), min=0.0)
    put_payoff = torch.clamp(float(K) - S_T, min=0.0)
    discount = float(np.exp(-float(r) * float(T)))

    prices = {
        "van_call": discount * call_payoff.mean().item(),
        "van_put": discount * put_payoff.mean().item(),
        "barr_call": np.nan,
        "barr_put": np.nan,
    }

    if B is not None:
        if samples.shape[1] < 2:
            raise ValueError(
                "Barrier pricing requires CVAE output [X_T, M_T]."
            )
        M_T = samples[:, 1]
        alive = (
            float(S0) * torch.exp(M_T) > float(B)
        ).to(dtype=call_payoff.dtype)

        prices["barr_call"] = discount * (call_payoff * alive).mean().item()
        prices["barr_put"] = discount * (put_payoff * alive).mean().item()

    return prices


def _default_steps(
    eta,
    model_type,
    relative_step,
):
    if relative_step <= 0:
        raise ValueError("relative_step must be positive.")

    if model_type == "bs":
        S0, _, r, sigma, T = eta
        values = {
            "delta": S0,
            "vega": sigma,
            "rho": r,
            "theta": T,
        }
    else:
        S0, _, r, _, _, _, _, v0, T = eta
        values = {
            "delta": S0,
            "v0": v0,
            "rho": r,
            "theta": T,
        }

    return {
        name: max(abs(float(value)) * relative_step, 1e-6)
        for name, value in values.items()
    }


def _parameter_spec(model_type):
    if model_type == "bs":
        return {
            "delta": (0, False, 1.0),
            "vega": (3, True, 1.0),
            "rho": (2, True, 1.0),
            "theta": (4, True, -1.0),
        }

    return {
        "delta": (0, False, 1.0),
        "rho": (2, True, 1.0),
        "v0": (7, True, 1.0),
        "theta": (8, True, -1.0),
    }


@torch.no_grad()
def CVAE_greeks(
    cvae,
    checkpoint,
    eta,
    model_type,
    B = 0.8,
    greeks = None,
    n_samples = 100000,
    h = None,
    relative_step = 0.01,
    antithetic = True,
    seed = 1234,
    batch_size = 65536,
    check_training_bounds = True,
):
    aliases = {
        "bs": "bs",
        "bs_clip": "bs",
        "hes": "hes",
        "heston": "hes",
        "hes_clip": "hes",
    }
    if model_type not in aliases:
        raise ValueError(
            "model_type must be 'bs', 'bs_clip', 'hes', 'heston', or 'hes_clip'."
        )
    backend = aliases[model_type]

    eta = np.asarray(eta, dtype=np.float64)
    expected_length = 5 if backend == "bs" else 9
    if eta.shape != (expected_length,):
        raise ValueError(
            f"{backend} eta must contain {expected_length} values, "
            f"got shape {eta.shape}."
        )

    eta_min = _as_numpy(checkpoint["eta_min"]).astype(np.float64)
    eta_max = _as_numpy(checkpoint["eta_max"]).astype(np.float64)

    eta_raw = eta[2:]
    eta_scaled = _normalize_eta(
        eta_raw,
        eta_min,
        eta_max,
        check_bounds=check_training_bounds,
        parameter_name="base",
    )

    spec = _parameter_spec(backend)
    if greeks is None:
        greeks = (
            ("delta", "vega", "rho", "theta")
            if backend == "bs"
            else ("delta", "rho", "v0", "theta")
        )
    greeks = tuple(greeks)

    unknown = set(greeks) - (set(spec))
    if unknown:
        raise ValueError(
            f"Unsupported Greeks for {backend}: {sorted(unknown)}. "
            f"Available: {sorted(spec)}."
        )

    steps = _default_steps(eta, backend, relative_step)
    if h is not None:
        unknown_h = set(h) - set(spec)
        if unknown_h:
            raise ValueError(f"Unknown h keys: {sorted(unknown_h)}.")
        for name, value in h.items():
            value = float(value)
            if value <= 0:
                raise ValueError(f"h['{name}'] must be positive.")
            steps[name] = value

    eps_z, eps_x = _make_common_noise(
        cvae,
        int(n_samples),
        antithetic=bool(antithetic),
        seed=seed,
    )

    base_samples = sample_cvae_with_noise(
        cvae, eta_scaled, eps_z, eps_x, batch_size=batch_size,
    )
    base_prices = prices_from_cvae_samples(
        base_samples,
        S0=float(eta[0]),
        K=float(eta[1]),
        B=B,
        r=float(eta[2]),
        T=float(eta[-1]),
    )

    sample_cache: dict[tuple[float, ...], torch.Tensor] = {
        tuple(eta[2:].tolist()): base_samples,
    }

    def prices_for(current_eta):
        condition_key = tuple(current_eta[2:].tolist())
        if condition_key not in sample_cache:
            scaled = _normalize_eta(
                current_eta[2:],
                eta_min,
                eta_max,
                check_bounds=check_training_bounds,
                parameter_name="bump",
            )
            sample_cache[condition_key] = sample_cvae_with_noise(
                cvae, scaled, eps_z, eps_x, batch_size=batch_size
            )
        return prices_from_cvae_samples(
            sample_cache[condition_key],
            S0=float(current_eta[0]),
            K=float(current_eta[1]),
            B=B,
            r=float(current_eta[2]),
            T=float(current_eta[-1]),
        )

    def bump_is_valid(index, step, lower, upper):
        value = eta[index] + step
        if not lower < value < upper:
            return False
        if index == 0 and B is not None and value <= B:
            return False
        if check_training_bounds and index >= 2:
            condition = eta[2:].copy()
            condition[index - 2] = value
            return bool(np.all(condition >= eta_min) and np.all(condition <= eta_max))
        return True

    bounds = {
        "delta": (0.0, np.inf),
        "vega": (0.0, np.inf),
        "rho": (-np.inf, np.inf),
        "theta": (0.0, np.inf),
        "v0": (0.0, np.inf),
    }
    greek_values: dict[str, dict[str, float]] = {}
    details: dict[str, dict[str, Any]] = {}

    for greek_name in greeks:
        full_index, _, sign = spec[greek_name]
        step = steps[greek_name]
        lower, upper = bounds[greek_name]
        if not bump_is_valid(full_index, step, lower, upper):
            raise ValueError(f"No valid forward bump for {greek_name}.")

        eta_up = eta.copy()
        eta_up[full_index] += step
        up_prices = prices_for(eta_up)
        values = {
            key: np.nan if not np.isfinite(up_prices[key]) else
            sign * (up_prices[key] - base_prices[key]) / step 
            for key in OPTION_KEYS
        }
        greek_values[greek_name] = values
        details[greek_name] = {
            "h": step,
            "scheme": "forward",
            "parameter": greek_name,
        }

    return {
        "model_type": backend,
        "antithetic": bool(antithetic),
        "n_samples": int(n_samples),
        "seed": seed,
        "base_eta": eta.copy(),
        "base_prices": base_prices,
        "h": {name: steps[name] for name in greeks},
        "greeks": greek_values,
        "details": details,
    }