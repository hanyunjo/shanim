import math

import torch


OPTION_KEYS = ("van_call", "van_put", "barr_call", "barr_put")


def _model_device_dtype(cvae):
    parameter = next(cvae.parameters())
    return parameter.device, parameter.dtype


def _resolve_model_type(model_type):
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
    return aliases[model_type]


def _make_common_noise(
    cvae,
    n_samples,
    antithetic,
    seed,
):
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if antithetic and n_samples % 2 != 0:
        raise ValueError("n_samples must be even when antithetic=True.")

    device, dtype = _model_device_dtype(cvae)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))

    n_base = n_samples // 2 if antithetic else n_samples
    eps_z_base = torch.randn(
        (n_base, cvae.dim_z),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    eps_x_base = torch.randn(
        (n_base, cvae.dim_x),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    if not antithetic:
        return eps_z_base, eps_x_base

    return (
        torch.cat((eps_z_base, -eps_z_base), dim=0),
        torch.cat((eps_x_base, -eps_x_base), dim=0),
    )


def _parameter_spec(model_type):
    if model_type == "bs":
        return {
            "delta": (0, 1.0),
            "rho": (2, 1.0),
            "vega": (3, 1.0),
            "theta": (4, -1.0),
        }

    return {
        "delta": (0, 1.0),
        "rho": (2, 1.0),
        "kappa": (3, 1.0),
        "long_var": (4, 1.0),
        "xi": (5, 1.0),
        "corr": (6, 1.0),
        "v0": (7, 1.0),
        "theta": (8, -1.0),
    }


def _prepare_checkpoint_bounds(
    checkpoint,
    device,
    dtype,
):
    eta_min = torch.as_tensor(
        checkpoint["eta_min"],
        device=device,
        dtype=dtype,
    ).reshape(-1)
    eta_max = torch.as_tensor(
        checkpoint["eta_max"],
        device=device,
        dtype=dtype,
    ).reshape(-1)

    if not bool(torch.isfinite(eta_min).all() and torch.isfinite(eta_max).all()):
        raise ValueError("checkpoint eta_min and eta_max must be finite.")
    if bool((eta_max <= eta_min).any()):
        raise ValueError("checkpoint eta_max must be greater than eta_min.")

    return eta_min, eta_max


@torch.enable_grad()
def CVAE_greeks_AD(
    cvae,
    checkpoint,
    eta,
    model_type,
    B = 0.8,
    barrier_mode = "hard",
    sigmoid_tau = 0.01,
    greeks = None,
    n_samples = 100000,
    antithetic = True,
    seed = 1234,
    batch_size = 16384,
    check_training_bounds = True,
):
    """Estimate CVAE prices and first-order Greeks by automatic differentiation.

    ``barrier_mode='hard'`` differentiates through the payoff while treating the
    barrier-survival indicator as fixed. ``'sigmoid'`` instead differentiates a
    smooth approximation of the barrier indicator. Set ``B=None`` for vanilla
    options only; barrier prices and Greeks are then returned as ``nan``.
    """
    backend = _resolve_model_type(model_type)
    if barrier_mode not in ("hard", "sigmoid"):
        raise ValueError("barrier_mode must be 'hard' or 'sigmoid'.")

    sigmoid_tau = float(sigmoid_tau)
    if sigmoid_tau <= 0:
        raise ValueError("sigmoid_tau must be positive.")

    n_samples = int(n_samples)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    device, dtype = _model_device_dtype(cvae)
    eta_tensor = torch.as_tensor(eta, device=device, dtype=dtype).reshape(-1)
    expected_eta_len = 5 if backend == "bs" else 9
    if eta_tensor.numel() != expected_eta_len:
        raise ValueError(
            f"{backend} eta must contain {expected_eta_len} values, "
            f"got {eta_tensor.numel()}."
        )
    if not bool(torch.isfinite(eta_tensor).all()):
        raise ValueError("eta must contain only finite values.")

    eta_values = eta_tensor.detach()
    S0_value = float(eta_values[0].item())
    K_value = float(eta_values[1].item())
    T_value = float(eta_values[-1].item())
    if S0_value <= 0 or K_value <= 0 or T_value <= 0:
        raise ValueError("S0, K, and T must be positive.")
    if cvae.dim_x < 1:
        raise ValueError("CVAE output must include X_T in column 0.")

    B_value = None if B is None else float(B)
    if B_value is not None:
        if not math.isfinite(B_value) or not 0 < B_value < S0_value:
            raise ValueError("A down barrier must satisfy 0 < B < S0.")
        if cvae.dim_x < 2:
            raise ValueError("Barrier pricing requires CVAE output [X_T, M_T].")

    eta_min, eta_max = _prepare_checkpoint_bounds(
        checkpoint,
        device,
        dtype,
    )
    if eta_min.numel() != cvae.dim_eta or eta_max.numel() != cvae.dim_eta:
        raise ValueError(
            "checkpoint eta_min/eta_max do not match cvae.dim_eta: "
            f"{eta_min.numel()}, {eta_max.numel()} vs {cvae.dim_eta}."
        )

    condition_values = eta_values[2:]
    if check_training_bounds:
        outside = (condition_values < eta_min) | (condition_values > eta_max)
        if bool(outside.any()):
            invalid = torch.nonzero(outside, as_tuple=False).reshape(-1).tolist()
            raise ValueError(
                "eta is outside the CVAE training range at conditional indices "
                f"{invalid}."
            )

    spec = _parameter_spec(backend)
    if greeks is None:
        greeks = (
            ("delta", "vega", "rho", "theta")
            if backend == "bs"
            else ("delta", "rho", "v0", "theta")
        )
    greeks = tuple(greeks)
    unknown = set(greeks) - set(spec)
    if unknown:
        raise ValueError(
            f"Unsupported Greeks for {backend}: {sorted(unknown)}. "
            f"Available: {sorted(spec)}."
        )

    eps_z, eps_x = _make_common_noise(
        cvae,
        n_samples,
        antithetic=bool(antithetic),
        seed=seed,
    )
    active_option_keys = OPTION_KEYS if B_value is not None else OPTION_KEYS[:2]

    # Model parameters stay fixed; only eta participates in autograd.
    parameters = list(cvae.parameters())
    old_requires_grad = [parameter.requires_grad for parameter in parameters]
    old_training = cvae.training
    cvae.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)

    eta_leaf = eta_tensor.detach().clone().requires_grad_(True)
    total_prices = {
        key: 0.0 if key in active_option_keys else float("nan")
        for key in OPTION_KEYS
    }
    total_gradients = {
        key: torch.zeros_like(eta_leaf) for key in OPTION_KEYS
    }

    try:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            n_batch = end - start
            sample_weight = n_batch / n_samples

            eta_scaled = (eta_leaf[2:] - eta_min) / (eta_max - eta_min + 1e-8)
            eta_batch = eta_scaled.unsqueeze(0).expand(n_batch, -1)

            mu_p, lv_p = cvae.prior(eta_batch)
            z = cvae.reparameterize(mu_p, lv_p, eps_z[start:end])
            mu_x, lv_x = cvae.decoder(z, eta_batch)
            samples = cvae.reparameterize(mu_x, lv_x, eps_x[start:end])

            S0 = eta_leaf[0]
            K = eta_leaf[1]
            r = eta_leaf[2]
            T = eta_leaf[-1]
            S_T = S0 * torch.exp(samples[:, 0])
            discount = torch.exp(-r * T)
            call_payoff = torch.relu(S_T - K)
            put_payoff = torch.relu(K - S_T)

            batch_prices = {
                "van_call": sample_weight * discount * call_payoff.mean(),
                "van_put": sample_weight * discount * put_payoff.mean(),
            }
            if B_value is not None:
                M_T = samples[:, 1]
                if barrier_mode == "hard":
                    alive = (S0 * torch.exp(M_T) > B_value).to(dtype=dtype)
                else:
                    margin = torch.log(S0 / B_value) + M_T
                    alive = torch.sigmoid(margin / sigmoid_tau)
                batch_prices["barr_call"] = (
                    sample_weight * discount * (call_payoff * alive).mean()
                )
                batch_prices["barr_put"] = (
                    sample_weight * discount * (put_payoff * alive).mean()
                )

            price_items = tuple(batch_prices.items())
            for item_index, (option_key, price_tensor) in enumerate(price_items):
                gradient = torch.autograd.grad(
                    price_tensor,
                    eta_leaf,
                    retain_graph=item_index < len(price_items) - 1,
                    create_graph=False,
                )[0]
                total_gradients[option_key] += gradient.detach()
                total_prices[option_key] += float(price_tensor.detach().item())
    finally:
        for parameter, flag in zip(parameters, old_requires_grad):
            parameter.requires_grad_(flag)
        cvae.train(old_training)

    greek_values = {}
    for greek_name in greeks:
        eta_index, sign = spec[greek_name]
        greek_values[greek_name] = {
            option_key: (
                float("nan")
                if option_key not in active_option_keys
                else float(sign * total_gradients[option_key][eta_index].item())
            )
            for option_key in OPTION_KEYS
        }

    details = {
        greek_name: {
            "scheme": "autodiff",
            "parameter": greek_name,
        }
        for greek_name in greeks
    }
    warning = None
    if B_value is not None and barrier_mode == "hard":
        warning = "Hard-indicator AD ignores the derivative of barrier survival."
    elif B_value is not None:
        warning = "Sigmoid AD estimates Greeks of a smoothed barrier payoff."

    return {
        "model_type": backend,
        "barrier_mode": barrier_mode if B_value is not None else None,
        "sigmoid_tau": sigmoid_tau if B_value is not None and barrier_mode == "sigmoid" else None,
        "antithetic": bool(antithetic),
        "n_samples": n_samples,
        "seed": seed,
        "base_eta": eta_values.cpu().numpy(),
        "base_prices": total_prices,
        "greeks": greek_values,
        "details": details,
        "raw_eta_gradients": {
            option_key: total_gradients[option_key].cpu().numpy()
            for option_key in OPTION_KEYS
        },
        "warning": warning,
    }


def compare_hard_and_sigmoid_AD(
    cvae,
    checkpoint,
    eta,
    model_type,
    B = 0.8,
    sigmoid_tau = 0.01,
    greeks = None,
    n_samples = 100000,
    antithetic = True,
    seed = 1234,
    batch_size = 16384,
    check_training_bounds = True,
):
    """Compare hard-indicator and sigmoid-smoothed AD using common noise."""
    common_kwargs = {
        "model_type": model_type,
        "B": B,
        "greeks": greeks,
        "n_samples": n_samples,
        "antithetic": antithetic,
        "seed": seed,
        "batch_size": batch_size,
        "check_training_bounds": check_training_bounds,
    }
    return {
        "hard": CVAE_greeks_AD(
            cvae,
            checkpoint,
            eta,
            barrier_mode="hard",
            **common_kwargs,
        ),
        "sigmoid": CVAE_greeks_AD(
            cvae,
            checkpoint,
            eta,
            barrier_mode="sigmoid",
            sigmoid_tau=sigmoid_tau,
            **common_kwargs,
        ),
    }
