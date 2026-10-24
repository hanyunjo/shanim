from __future__ import annotations

import secrets
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


OPTION_KEYS = ("van_call", "van_put", "barr_call", "barr_put")


def _normalize_tail_multiple(multiple):
    if isinstance(multiple, (bool, np.bool_)) or not isinstance(
        multiple, (int, float, np.integer, np.floating)
    ):
        raise TypeError("multiple must be a real number")
    multiple = float(multiple)
    if not np.isfinite(multiple) or multiple < 0 or (0 < multiple <= 1):
        raise ValueError("multiple must be 0 (disabled) or greater than 1")
    return multiple


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
    if seed is None:
        generator.seed()
    else:
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


@torch.no_grad()
def sample_cvae_with_tail_resampling(
    cvae,
    eta_scaled,
    eps_z,
    eps_x,
    validate_samples=False,
    valid_resample=False,
    multiple=0,
    batch_size=65536,
    antithetic=True,
    seed=1234,
    max_resample_rounds=100,
):
    """Validate/refill samples and optionally apply the X_T tail cutoff."""
    if not isinstance(validate_samples, (bool, np.bool_)):
        raise TypeError("validate_samples must be True or False")
    if not isinstance(valid_resample, (bool, np.bool_)):
        raise TypeError("valid_resample must be True or False")

    validation_applied = bool(validate_samples and valid_resample)
    multiple = _normalize_tail_multiple(multiple)
    tail_filter_applied = multiple > 0
    effective_seed = secrets.randbits(63) if seed is None else int(seed)
    max_resample_rounds = int(max_resample_rounds)
    if max_resample_rounds < 1:
        raise ValueError("max_resample_rounds must be at least 1")

    requested = int(eps_z.shape[0])
    if not validation_applied and not tail_filter_applied:
        samples = sample_cvae_with_noise(
            cvae, eta_scaled, eps_z, eps_x, batch_size=batch_size
        )
        return samples, {
            "validation_applied": False,
            "valid_resample": False,
            "valid_resample_requested": bool(valid_resample),
            "support_invalid_count": None,
            "invalid_path_fraction": None,
            "rejected_sample_count": 0,
            "seed": effective_seed,
            "draw_count_by_round": [requested],
            "antithetic_input": bool(antithetic),
            "antithetic_retained": bool(antithetic),
            "tail_filter_applied": False,
            "multiple": multiple,
            "X_T_upper": None,
            "requested_sample_count": requested,
            "used_sample_count": requested,
            "total_drawn_sample_count": requested,
            "tail_rejected_sample_count": 0,
            "tail_rejected_sample_fraction": 0.0,
            "resample_round_count": 0,
        }

    xt_upper = float(np.log(multiple)) if tail_filter_applied else None
    accepted_batches = []
    accepted_count = 0
    total_drawn = 0
    rejected = 0
    support_invalid_count = 0
    tail_rejected = 0
    rounds_used = 0
    draw_count_by_round = []

    for round_index in range(max_resample_rounds):
        remaining = requested - accepted_count
        if remaining <= 0:
            break
        rounds_used = round_index + 1

        if round_index == 0:
            round_eps_z, round_eps_x = eps_z, eps_x
        else:
            # A condition-independent draw size keeps the refill noise stream
            # common across the base and every bumped parameter value.
            draw_count = requested
            round_seed = effective_seed + round_index * 1_000_003
            round_eps_z, round_eps_x = _make_common_noise(
                cvae,
                draw_count,
                antithetic=bool(antithetic),
                seed=round_seed,
            )

        candidates = sample_cvae_with_noise(
            cvae,
            eta_scaled,
            round_eps_z,
            round_eps_x,
            batch_size=batch_size,
        )
        draw_count_by_round.append(int(candidates.shape[0]))
        finite = torch.isfinite(candidates).all(dim=1)
        if validation_applied:
            if candidates.shape[1] >= 2:
                X_T = candidates[:, 0]
                M_T = candidates[:, 1]
                support_invalid = (~finite) | (
                    M_T > torch.minimum(torch.zeros_like(X_T), X_T) + 1e-6
                )
            else:
                support_invalid = ~finite
        else:
            support_invalid = torch.zeros_like(finite)

        if tail_filter_applied:
            X_T = candidates[:, 0]
            tail_invalid = (~torch.isfinite(X_T)) | (X_T >= xt_upper)
        else:
            tail_invalid = torch.zeros_like(finite)

        invalid = support_invalid | tail_invalid
        valid = ~invalid
        accepted = candidates[valid][:remaining]

        total_drawn += int(candidates.shape[0])
        rejected += int(invalid.sum().item())
        support_invalid_count += int(support_invalid.sum().item())
        tail_rejected += int(tail_invalid.sum().item())
        if accepted.shape[0]:
            accepted_batches.append(accepted)
            accepted_count += int(accepted.shape[0])

    if accepted_count < requested:
        criteria = []
        if validation_applied:
            criteria.append("valid CVAE support")
        if tail_filter_applied:
            criteria.append(f"X_T < log({multiple:g})")
        raise RuntimeError(
            f"Could obtain only {accepted_count:,}/{requested:,} samples "
            f"satisfying {' and '.join(criteria)} after "
            f"{max_resample_rounds} rounds."
        )

    samples = torch.cat(accepted_batches, dim=0)[:requested]
    return samples, {
        "validation_applied": validation_applied,
        "valid_resample": validation_applied,
        "valid_resample_requested": bool(valid_resample),
        "support_invalid_count": (
            support_invalid_count if validation_applied else None
        ),
        "invalid_path_fraction": rejected / total_drawn,
        "rejected_sample_count": rejected,
        "seed": effective_seed,
        "draw_count_by_round": draw_count_by_round,
        "antithetic_input": bool(antithetic),
        "antithetic_retained": bool(antithetic and rejected == 0),
        "tail_filter_applied": tail_filter_applied,
        "multiple": multiple,
        "X_T_upper": xt_upper,
        "requested_sample_count": requested,
        "used_sample_count": int(samples.shape[0]),
        "total_drawn_sample_count": total_drawn,
        "tail_rejected_sample_count": tail_rejected,
        "tail_rejected_sample_fraction": tail_rejected / total_drawn,
        "resample_round_count": max(rounds_used - 1, 0),
    }


def prices_from_cvae_samples(
    samples,
    S0,
    K,
    B,
    r,
    T,
    sigma = None,
    dt = 0.001,
    bbc_correction = True,
    mt_corr = False,
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
        if not isinstance(mt_corr, (bool, np.bool_)):
            raise TypeError("mt_corr must be True or False.")
        if mt_corr:
            M_T = torch.minimum(
                M_T,
                torch.minimum(torch.zeros_like(X_T), X_T),
            )

        barrier_log = float(np.log(float(B) / float(S0)))
        discrete_alive = M_T > barrier_log
        survival = discrete_alive.to(dtype=call_payoff.dtype)

        if not isinstance(bbc_correction, (bool, np.bool_)):
            raise TypeError("bbc_correction must be True or False.")
        if samples.shape[1] >= 3 and bbc_correction:
            dt = float(dt)
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("dt must be a positive finite number for BBC pricing.")

            M_next_raw = samples[:, 2]
            finite_pair = torch.isfinite(M_T) & torch.isfinite(M_next_raw)
            M_next = torch.maximum(M_next_raw, M_T)

            if samples.shape[1] == 3:
                if sigma is None:
                    raise ValueError(
                        "dim_x=3 BBC pricing requires the raw BS sigma."
                    )
                sigma = float(sigma)
                if not np.isfinite(sigma) or sigma <= 0.0:
                    raise ValueError("sigma must be a positive finite number.")
                local_variance = torch.full_like(M_T, sigma ** 2)
            else:
                v_avg_raw = samples[:, 3]
                finite_variance = torch.isfinite(v_avg_raw)
                local_variance = torch.nan_to_num(
                    v_avg_raw, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp_min(0.0)
                finite_pair = finite_pair & finite_variance

            endpoint_alive = finite_pair & discrete_alive & (M_next > barrier_log)
            distance_min = (M_T - barrier_log).clamp_min(0.0)
            distance_next = (M_next - barrier_log).clamp_min(0.0)
            denominator = local_variance * dt + 1e-10
            exponent = -2.0 * distance_min * distance_next / denominator
            p_cross = torch.where(
                endpoint_alive,
                torch.exp(torch.clamp(exponent, max=0.0)),
                torch.zeros_like(M_T),
            ).clamp_(0.0, 1.0)
            survival = torch.where(
                endpoint_alive,
                1.0 - p_cross,
                torch.zeros_like(M_T),
            )

        prices["barr_call"] = discount * (call_payoff * survival).mean().item()
        prices["barr_put"] = discount * (put_payoff * survival).mean().item()

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


def _resolve_model_type(model_type):
    aliases = {
        "bs": "bs",
        "bs_clip": "bs",
        "bs_bbc": "bs",
        "bs_ori": "bs",
        "hes": "hes",
        "heston": "hes",
        "hes_clip": "hes",
        "hes_bbc": "hes",
        "hes_ori": "hes",
    }
    try:
        return aliases[model_type]
    except KeyError as exc:
        raise ValueError(
            f"model_type must be one of {sorted(aliases)}."
        ) from exc


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
    dt = 0.001,
    bbc_correction = True,
    mt_corr = False,
    validate_samples = False,
    valid_resample = False,
    multiple = 0,
    max_resample_rounds = 100,
):
    backend = _resolve_model_type(model_type)
    requested_seed = seed
    seed = secrets.randbits(63) if seed is None else int(seed)
    expected_bbc_dims = {"bs_bbc": 3, "hes_bbc": 4}
    if model_type in expected_bbc_dims and cvae.dim_x != expected_bbc_dims[model_type]:
        raise ValueError(
            f"model_type='{model_type}' requires dim_x="
            f"{expected_bbc_dims[model_type]}, got dim_x={cvae.dim_x}."
        )

    eta = np.asarray(eta, dtype=np.float64)
    expected_length = 5 if backend == "bs" else 9
    if eta.shape != (expected_length,):
        raise ValueError(
            f"{backend} eta must contain {expected_length} values, "
            f"got shape {eta.shape}."
        )
    multiple = _normalize_tail_multiple(multiple)

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

    base_samples, base_sampling_diagnostics = sample_cvae_with_tail_resampling(
        cvae,
        eta_scaled,
        eps_z,
        eps_x,
        validate_samples=validate_samples,
        valid_resample=valid_resample,
        multiple=multiple,
        batch_size=batch_size,
        antithetic=bool(antithetic),
        seed=seed,
        max_resample_rounds=max_resample_rounds,
    )
    base_prices = prices_from_cvae_samples(
        base_samples,
        S0=float(eta[0]),
        K=float(eta[1]),
        B=B,
        r=float(eta[2]),
        T=float(eta[-1]),
        sigma=float(eta[3]) if backend == "bs" else None,
        dt=dt,
        bbc_correction=bbc_correction,
        mt_corr=mt_corr,
    )

    sample_cache: dict[tuple[float, ...], torch.Tensor] = {
        tuple(eta[2:].tolist()): base_samples,
    }
    sampling_diagnostics_cache: dict[tuple[float, ...], dict[str, Any]] = {
        tuple(eta[2:].tolist()): base_sampling_diagnostics,
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
            (
                sample_cache[condition_key],
                sampling_diagnostics_cache[condition_key],
            ) = sample_cvae_with_tail_resampling(
                cvae,
                scaled,
                eps_z,
                eps_x,
                validate_samples=validate_samples,
                valid_resample=valid_resample,
                multiple=multiple,
                batch_size=batch_size,
                antithetic=bool(antithetic),
                seed=seed,
                max_resample_rounds=max_resample_rounds,
            )
        return prices_from_cvae_samples(
            sample_cache[condition_key],
            S0=float(current_eta[0]),
            K=float(current_eta[1]),
            B=B,
            r=float(current_eta[2]),
            T=float(current_eta[-1]),
            sigma=float(current_eta[3]) if backend == "bs" else None,
            dt=dt,
            bbc_correction=bbc_correction,
            mt_corr=mt_corr,
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
            "sampling_diagnostics": sampling_diagnostics_cache[
                tuple(eta_up[2:].tolist())
            ],
        }

    return {
        "model_type": backend,
        "model_variant": model_type,
        "antithetic": bool(antithetic),
        "n_samples": int(n_samples),
        "seed": seed,
        "requested_seed": requested_seed,
        "bbc_correction": bool(bbc_correction),
        "bbc_dt": float(dt) if B is not None and cvae.dim_x >= 3 else None,
        "mt_corr": bool(mt_corr),
        "validate_samples": bool(validate_samples),
        "valid_resample": bool(valid_resample),
        "multiple": multiple,
        "X_T_upper": base_sampling_diagnostics["X_T_upper"],
        "sampling_diagnostics": base_sampling_diagnostics,
        "base_eta": eta.copy(),
        "base_prices": base_prices,
        "h": {name: steps[name] for name in greeks},
        "greeks": greek_values,
        "details": details,
    }
