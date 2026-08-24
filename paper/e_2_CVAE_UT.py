import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")

SUPPORT_EPS = 1e-6
SUPPORT_TOLERANCE = 1e-6
UT_MODEL_COORDINATES = ["X_T", "U_T"]
DIRECT_MODEL_COORDINATES = ["X_T", "M_T"]
PHYSICAL_COORDINATES = ["X_T", "M_T"]
SUPPORT_PARAMETERIZATION = "M_T = min(0, X_T) - softplus(U_T)"


# ─────────────────────────────
# Running-minimum transformations
# ─────────────────────────────
def inverse_softplus_tensor(y):
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if not torch.is_floating_point(y):
        y = y.to(torch.get_default_dtype())
    if not torch.isfinite(y).all():
        raise ValueError("inverse_softplus_tensor requires finite y.")
    if torch.any(y <= 0):
        raise ValueError("inverse_softplus_tensor requires y > 0.")
    return y + torch.log(-torch.expm1(-y))


def inverse_softplus_numpy(y):
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float64)
    if not np.isfinite(y).all():
        raise ValueError("inverse_softplus_numpy requires finite y.")
    if np.any(y <= 0):
        raise ValueError("inverse_softplus_numpy requires y > 0.")
    return y + np.log(-np.expm1(-y))


def _validate_running_minimum_tensor(X_T, M_T, tolerance=SUPPORT_TOLERANCE):
    if X_T.shape != M_T.shape:
        raise ValueError("X_T and M_T must have the same shape.")
    finite = torch.isfinite(X_T) & torch.isfinite(M_T)
    upper = torch.minimum(torch.zeros_like(X_T), X_T)
    invalid = (~finite) | (M_T > upper + tolerance)
    invalid_count = int(invalid.sum().item())
    if invalid_count:
        finite_violation = torch.where(finite, M_T - upper, torch.zeros_like(M_T))
        max_violation = float(finite_violation.max().item())
        raise ValueError(
            "Raw training data violates M_T <= min(0, X_T) + tolerance: "
            f"invalid={invalid_count:,}/{X_T.numel():,}, "
            f"max_violation={max_violation:.6g}, tolerance={tolerance:.6g}. "
            "The raw samples were not corrected or dropped."
        )


def _validate_running_minimum_numpy(X_T, M_T, tolerance=SUPPORT_TOLERANCE):
    X_T = np.asarray(X_T)
    M_T = np.asarray(M_T)
    if X_T.shape != M_T.shape:
        raise ValueError("X_T and M_T must have the same shape.")
    finite = np.isfinite(X_T) & np.isfinite(M_T)
    upper = np.minimum(0.0, X_T)
    invalid = (~finite) | (M_T > upper + tolerance)
    invalid_count = int(invalid.sum())
    if invalid_count:
        finite_violation = np.where(finite, M_T - upper, 0.0)
        max_violation = float(np.max(finite_violation))
        raise ValueError(
            "Raw training data violates M_T <= min(0, X_T) + tolerance: "
            f"invalid={invalid_count:,}/{X_T.size:,}, "
            f"max_violation={max_violation:.6g}, tolerance={tolerance:.6g}. "
            "The raw samples were not corrected or dropped."
        )


def running_min_to_u_tensor(X_T, M_T, eps=SUPPORT_EPS,
                            tolerance=SUPPORT_TOLERANCE):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    _validate_running_minimum_tensor(X_T, M_T, tolerance=tolerance)
    D_T = torch.minimum(torch.zeros_like(X_T), X_T) - M_T
    D_safe = torch.clamp(D_T, min=eps)
    return inverse_softplus_tensor(D_safe)


def running_min_to_u_numpy(X_T, M_T, eps=SUPPORT_EPS,
                           tolerance=SUPPORT_TOLERANCE):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    X_T = np.asarray(X_T)
    M_T = np.asarray(M_T)
    _validate_running_minimum_numpy(X_T, M_T, tolerance=tolerance)
    D_T = np.minimum(0.0, X_T) - M_T
    D_safe = np.maximum(D_T, eps)
    return inverse_softplus_numpy(D_safe)


def xu_to_running_min_tensor(X_T, U_T):
    if X_T.shape != U_T.shape:
        raise ValueError("X_T and U_T must have the same shape.")
    D_T = F.softplus(U_T)
    M_T = torch.minimum(torch.zeros_like(X_T), X_T) - D_T
    return M_T


def xu_to_running_min_numpy(X_T, U_T):
    X_T = np.asarray(X_T)
    U_T = np.asarray(U_T)
    if X_T.shape != U_T.shape:
        raise ValueError("X_T and U_T must have the same shape.")
    D_T = np.logaddexp(0.0, U_T)
    return np.minimum(0.0, X_T) - D_T


def physical_to_model_tensor(x_physical, eps=SUPPORT_EPS,
                             tolerance=SUPPORT_TOLERANCE):
    if x_physical.ndim < 2 or x_physical.shape[-1] != 2:
        raise ValueError("Expected raw samples with shape (..., 2) = [X_T, M_T].")
    X_T = x_physical[..., 0]
    M_T = x_physical[..., 1]
    U_T = running_min_to_u_tensor(
        X_T, M_T, eps=eps, tolerance=tolerance,
    )
    return torch.stack((X_T, U_T), dim=-1)


def physical_to_model_numpy(x_physical, eps=SUPPORT_EPS,
                            tolerance=SUPPORT_TOLERANCE):
    x_physical = np.asarray(x_physical)
    if x_physical.ndim < 2 or x_physical.shape[-1] != 2:
        raise ValueError("Expected raw samples with shape (..., 2) = [X_T, M_T].")
    X_T = x_physical[..., 0]
    M_T = x_physical[..., 1]
    U_T = running_min_to_u_numpy(
        X_T, M_T, eps=eps, tolerance=tolerance,
    )
    return np.stack((X_T, U_T), axis=-1)


def model_to_physical_tensor(x_model):
    if x_model.ndim < 2 or x_model.shape[-1] != 2:
        raise ValueError("Expected model samples with shape (..., 2) = [X_T, U_T].")
    X_T = x_model[..., 0]
    U_T = x_model[..., 1]
    M_T = xu_to_running_min_tensor(X_T, U_T)
    return torch.stack((X_T, M_T), dim=-1)


def _finite_summary(values):
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return float("nan"), float("nan")
    return float(finite_values.min().item()), float(finite_values.max().item())


def _quantile_summary(values, probabilities):
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return {str(q): float("nan") for q in probabilities}
    q_tensor = torch.as_tensor(
        probabilities, device=finite_values.device, dtype=finite_values.dtype,
    )
    q_values = torch.quantile(finite_values, q_tensor)
    return {
        str(q): float(value.item())
        for q, value in zip(probabilities, q_values)
    }


def diagnose_cvae_samples(samples, transformed_samples=None,
                          tolerance=SUPPORT_TOLERANCE, print_output=True):
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape (N, 2) = [X_T, M_T].")
    if samples.shape[0] < 1:
        raise ValueError("samples must contain at least one row.")

    X_T = samples[:, 0]
    M_T = samples[:, 1]
    D_T = torch.minimum(torch.zeros_like(X_T), X_T) - M_T

    if transformed_samples is None:
        U_T = inverse_softplus_tensor(torch.clamp(D_T, min=SUPPORT_EPS))
    else:
        if transformed_samples.shape != samples.shape:
            raise ValueError("transformed_samples must have the same shape as samples.")
        if not torch.equal(transformed_samples[:, 0], X_T):
            if not torch.allclose(transformed_samples[:, 0], X_T):
                raise ValueError("X_T differs between physical and transformed samples.")
        U_T = transformed_samples[:, 1]

    finite_rows = (
        torch.isfinite(X_T) & torch.isfinite(M_T)
        & torch.isfinite(D_T) & torch.isfinite(U_T)
    )
    upper = torch.minimum(torch.zeros_like(X_T), X_T)
    invalid = finite_rows & (M_T > upper + tolerance)
    invalid_count = int(invalid.sum().item())
    sample_count = int(samples.shape[0])
    probabilities = [0.001, 0.01, 0.5, 0.99, 0.999]

    x_min, x_max = _finite_summary(X_T)
    m_min, m_max = _finite_summary(M_T)
    d_min, d_max = _finite_summary(D_T)
    u_min, u_max = _finite_summary(U_T)
    diagnostics = {
        "generated_sample_count": sample_count,
        "X_T_min": x_min,
        "X_T_max": x_max,
        "M_T_min": m_min,
        "M_T_max": m_max,
        "D_T_min": d_min,
        "D_T_max": d_max,
        "U_T_min": u_min,
        "U_T_max": u_max,
        "support_invalid_count": invalid_count,
        "support_invalid_fraction": invalid_count / sample_count,
        "finite_sample_fraction": float(finite_rows.float().mean().item()),
        "X_T_quantiles": _quantile_summary(X_T, probabilities),
        "M_T_quantiles": _quantile_summary(M_T, probabilities),
        "D_T_quantiles": _quantile_summary(D_T, probabilities),
        "support_tolerance": float(tolerance),
    }

    if print_output:
        print(f"generated sample count : {sample_count:,}")
        print(f"X_T min / max         : {x_min:.8g} / {x_max:.8g}")
        print(f"M_T min / max         : {m_min:.8g} / {m_max:.8g}")
        print(f"D_T min / max         : {d_min:.8g} / {d_max:.8g}")
        print(f"U_T min / max         : {u_min:.8g} / {u_max:.8g}")
        print(
            "support invalid        : "
            f"{invalid_count:,} ({100.0 * invalid_count / sample_count:.6f}%)"
        )
        print(
            "finite sample fraction : "
            f"{100.0 * diagnostics['finite_sample_fraction']:.6f}%"
        )
        print(f"X_T quantiles         : {diagnostics['X_T_quantiles']}")
        print(f"M_T quantiles         : {diagnostics['M_T_quantiles']}")
        print(f"D_T quantiles         : {diagnostics['D_T_quantiles']}")

    return diagnostics


# ────────────
# Sub-networks
# ────────────
def freeze_batchnorm(model, freeze_affine=True):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            if freeze_affine:
                if module.weight is not None:
                    module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.Tanh, use_bn=False):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.activation = activation()
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim) if use_bn else nn.Identity()

        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = self.linear1(x)
        residual = self.bn1(residual)
        residual = self.activation(residual)
        residual = self.linear2(residual)
        residual = self.bn2(residual)
        return x + residual


def _hidden_mlp(in_dim, hidden_dims, activation=nn.Tanh,
                use_bn=False, residual_blocks=0):
    if not isinstance(residual_blocks, (int, np.integer)) or isinstance(residual_blocks, bool):
        raise TypeError("residual_blocks must be a non-negative integer.")
    residual_blocks = int(residual_blocks)
    if residual_blocks < 0:
        raise ValueError("residual_blocks must be >= 0.")
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one hidden width.")

    eligible_blocks = 0
    layer_idx = 1
    prev_dim = hidden_dims[0]
    while layer_idx < len(hidden_dims):
        if (
            layer_idx + 1 < len(hidden_dims)
            and hidden_dims[layer_idx] == prev_dim
            and hidden_dims[layer_idx + 1] == prev_dim
        ):
            eligible_blocks += 1
            layer_idx += 2
        else:
            prev_dim = hidden_dims[layer_idx]
            layer_idx += 1

    if residual_blocks > eligible_blocks:
        raise ValueError(
            f"residual_blocks={residual_blocks}, but hidden_dims={hidden_dims} "
            f"has only {eligible_blocks} two-layer equal-width block(s)."
        )

    layers = [nn.Linear(in_dim, hidden_dims[0])]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
    layers.append(activation())

    prev_dim = hidden_dims[0]
    layer_idx = 1
    used_residual_blocks = 0
    while layer_idx < len(hidden_dims):
        use_residual = (
            used_residual_blocks < residual_blocks
            and layer_idx + 1 < len(hidden_dims)
            and hidden_dims[layer_idx] == prev_dim
            and hidden_dims[layer_idx + 1] == prev_dim
        )
        if use_residual:
            layers.append(ResidualBlock(prev_dim, activation=activation, use_bn=use_bn))
            used_residual_blocks += 1
            layer_idx += 2
        else:
            h_dim = hidden_dims[layer_idx]
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            prev_dim = h_dim
            layer_idx += 1

    return nn.Sequential(*layers)


class RecognitionNet(nn.Module):
    def __init__(self, dim_x, dim_eta, dim_z, hidden_dims, use_bn=False,
                 residual_blocks=0):
        super().__init__()
        self.net = _hidden_mlp(
            dim_x + dim_eta, hidden_dims, use_bn=use_bn,
            residual_blocks=residual_blocks,
        )
        self.mu = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, x, eta):
        h = self.net(torch.cat([x, eta], dim=-1))
        mu = self.mu(h)
        log_var = torch.clamp(self.logvar(h), min=-10.0, max=5.0)
        return mu, log_var


class PriorNet(nn.Module):
    def __init__(self, dim_eta, dim_z, hidden_dims, use_bn=False,
                 residual_blocks=0):
        super().__init__()
        self.net = _hidden_mlp(
            dim_eta, hidden_dims, use_bn=use_bn,
            residual_blocks=residual_blocks,
        )
        self.mu = nn.Linear(hidden_dims[-1], dim_z)
        self.logvar = nn.Linear(hidden_dims[-1], dim_z)

    def forward(self, eta):
        h = self.net(eta)
        mu = self.mu(h)
        log_var = torch.clamp(self.logvar(h), min=-10.0, max=5.0)
        return mu, log_var


class DecoderNet(nn.Module):
    def __init__(self, dim_z, dim_eta, dim_x, hidden_dims, use_bn=False,
                 residual_blocks=0):
        super().__init__()
        self.net = _hidden_mlp(
            dim_z + dim_eta, hidden_dims, use_bn=use_bn,
            residual_blocks=residual_blocks,
        )
        self.mu = nn.Linear(hidden_dims[-1], dim_x)
        self.logvar = nn.Linear(hidden_dims[-1], dim_x)

    def forward(self, z, eta):
        h = self.net(torch.cat([z, eta], dim=-1))
        mu = self.mu(h)
        log_var = torch.clamp(self.logvar(h), min=-10.0, max=5.0)
        return mu, log_var


# ─────────
# CVAE
# ─────────
class CVAE(nn.Module):
    RESIDUAL_LAYOUT = "paired_equal_width_v1"

    def __init__(self,
                 dim_x=2,
                 dim_eta=7,
                 dim_z=8,
                 hidden_dims=None,
                 use_bn=False,
                 residual_blocks=0,
                 support_eps=SUPPORT_EPS,
                 coordinate_system="transformed",
                 ):
        super().__init__()

        if dim_x != 2:
            raise ValueError(
                "The running-minimum UT model requires dim_x=2: [X_T, U_T]."
            )
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]
        if not isinstance(residual_blocks, (int, np.integer)) or isinstance(residual_blocks, bool):
            raise TypeError("residual_blocks must be a non-negative integer.")
        residual_blocks = int(residual_blocks)
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be >= 0.")
        if support_eps <= 0:
            raise ValueError("support_eps must be positive.")
        if coordinate_system not in ("transformed", "direct"):
            raise ValueError("coordinate_system must be 'transformed' or 'direct'.")

        self.dim_x = dim_x
        self.dim_eta = dim_eta
        self.dim_z = dim_z
        self.hidden_dims = list(hidden_dims)
        self.use_bn = bool(use_bn)
        self.residual_blocks = residual_blocks
        self.residual_layout = self.RESIDUAL_LAYOUT
        self.support_eps = float(support_eps)
        self.coordinate_system = coordinate_system
        self.model_coordinates = (
            list(UT_MODEL_COORDINATES)
            if coordinate_system == "transformed"
            else list(DIRECT_MODEL_COORDINATES)
        )
        self.physical_coordinates = list(PHYSICAL_COORDINATES)
        self.path_statistic = "running_minimum"
        self.last_sampling_diagnostics = None

        common_kwargs = {
            "hidden_dims": self.hidden_dims,
            "use_bn": self.use_bn,
            "residual_blocks": self.residual_blocks,
        }
        self.recognition = RecognitionNet(dim_x, dim_eta, dim_z, **common_kwargs)
        self.prior = PriorNet(dim_eta, dim_z, **common_kwargs)
        self.decoder = DecoderNet(dim_z, dim_eta, dim_x, **common_kwargs)

    def checkpoint_metadata(self):
        metadata = {
            "model_coordinates": list(self.model_coordinates),
            "physical_coordinates": list(self.physical_coordinates),
            "support_eps": self.support_eps,
            "path_statistic": self.path_statistic,
            "coordinate_system": self.coordinate_system,
        }
        if self.coordinate_system == "transformed":
            metadata["support_parameterization"] = SUPPORT_PARAMETERIZATION
        else:
            metadata["support_parameterization"] = "direct decoder output M_T"
        return metadata

    def get_extra_state(self):
        return self.checkpoint_metadata()

    def set_extra_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("CVAE checkpoint coordinate metadata is missing or invalid.")
        checkpoint_coordinates = state.get("model_coordinates")
        if checkpoint_coordinates != self.model_coordinates:
            raise ValueError(
                "Checkpoint/model coordinate mismatch: checkpoint uses "
                f"{checkpoint_coordinates}, current model expects {self.model_coordinates}. "
                "Use CVAE.from_checkpoint(...) to select the correct coordinate system."
            )
        if self.coordinate_system == "transformed":
            parameterization = state.get("support_parameterization")
            if parameterization != SUPPORT_PARAMETERIZATION:
                raise ValueError(
                    "The checkpoint has an incompatible support parameterization: "
                    f"{parameterization!r}."
                )
            checkpoint_eps = float(state.get("support_eps", float("nan")))
            if not np.isclose(checkpoint_eps, self.support_eps, rtol=0.0, atol=0.0):
                raise ValueError(
                    f"Checkpoint support_eps={checkpoint_eps}, "
                    f"current support_eps={self.support_eps}."
                )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if "_extra_state" not in state_dict:
            raise ValueError(
                "This state_dict has no CVAE coordinate metadata and is probably an old "
                "direct (X_T, M_T) checkpoint. Do not load it into the default UT model. "
                "Load the complete checkpoint with CVAE.from_checkpoint(...)."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location=None, strict=True,
                        **model_kwargs):
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            checkpoint = torch.load(
                checkpoint, map_location=map_location, weights_only=False,
            )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a checkpoint dict or path.")

        state_dict = checkpoint.get("model_state", checkpoint)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }
        else:
            state_dict = dict(state_dict)

        metadata = checkpoint
        if "model_coordinates" not in metadata:
            metadata = state_dict.get("_extra_state", {})
        coordinates = metadata.get("model_coordinates")
        if coordinates is None:
            coordinate_system = "direct"
            metadata = {
                "model_coordinates": list(DIRECT_MODEL_COORDINATES),
                "physical_coordinates": list(PHYSICAL_COORDINATES),
                "support_parameterization": "direct decoder output M_T",
                "support_eps": SUPPORT_EPS,
                "path_statistic": "running_minimum",
                "coordinate_system": "direct",
            }
        elif list(coordinates) == UT_MODEL_COORDINATES:
            coordinate_system = "transformed"
        elif list(coordinates) == DIRECT_MODEL_COORDINATES:
            coordinate_system = "direct"
        else:
            raise ValueError(f"Unsupported model_coordinates: {coordinates!r}")

        if "_extra_state" not in state_dict:
            state_dict["_extra_state"] = {
                "model_coordinates": list(metadata["model_coordinates"]),
                "physical_coordinates": list(
                    metadata.get("physical_coordinates", PHYSICAL_COORDINATES)
                ),
                "support_parameterization": metadata.get(
                    "support_parameterization",
                    SUPPORT_PARAMETERIZATION
                    if coordinate_system == "transformed"
                    else "direct decoder output M_T",
                ),
                "support_eps": float(metadata.get("support_eps", SUPPORT_EPS)),
                "path_statistic": metadata.get("path_statistic", "running_minimum"),
                "coordinate_system": coordinate_system,
            }

        constructor_keys = (
            "dim_x", "dim_eta", "dim_z", "hidden_dims", "use_bn",
            "residual_blocks", "support_eps",
        )
        if cls.__name__ == "CVAEBarrWeight":
            constructor_keys += (
                "weight_mode", "weight_alpha", "weight_mode2", "weight_alpha2",
                "weight_h", "weight_normalize", "S0", "K", "B", "cvae_type",
            )
        weight_config = checkpoint.get("weight_config", {})
        constructor_args = {}
        for key in constructor_keys:
            if key in checkpoint:
                constructor_args[key] = checkpoint[key]
            elif key in metadata:
                constructor_args[key] = metadata[key]
            elif key in weight_config:
                constructor_args[key] = weight_config[key]
        if coordinate_system == "transformed" and "support_eps" not in constructor_args:
            constructor_args["support_eps"] = metadata.get("support_eps", SUPPORT_EPS)
        constructor_args.update(model_kwargs)
        constructor_args["coordinate_system"] = coordinate_system
        model = cls(**constructor_args)
        model.load_state_dict(state_dict, strict=strict)
        return model

    def prepare_training_target(self, x_physical):
        if self.coordinate_system == "direct":
            if x_physical.ndim != 2 or x_physical.shape[-1] != 2:
                raise ValueError("Expected [X_T, M_T] with shape (batch, 2).")
            return x_physical
        return physical_to_model_tensor(
            x_physical, eps=self.support_eps,
            tolerance=SUPPORT_TOLERANCE,
        )

    @staticmethod
    def reparameterize(mu, log_var, eps=None):
        std = torch.exp(0.5 * log_var)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    def _elbo_components(self, x_physical, eta):
        x_model = self.prepare_training_target(x_physical)
        mu_q, lv_q = self.recognition(x_model, eta)
        mu_p, lv_p = self.prior(eta)
        z = self.reparameterize(mu_q, lv_q)
        mu_x, lv_x = self.decoder(z, eta)

        nll_per_sample = 0.5 * (
            lv_x + (x_model - mu_x).pow(2) / lv_x.exp()
            + np.log(2 * np.pi)
        ).sum(dim=-1)
        kl_dim_batch = 0.5 * (
            lv_p - lv_q
            + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp()
            - 1
        )
        return x_model, nll_per_sample, kl_dim_batch

    def forward(self, x, eta, return_kl_dim=False):
        _, nll_per_sample, kl_dim_batch = self._elbo_components(x, eta)
        recon_loss = nll_per_sample.mean()
        kl_dim_mean = kl_dim_batch.mean(dim=0)
        kl_loss = kl_dim_mean.sum()

        if return_kl_dim:
            return recon_loss, kl_loss, kl_dim_mean
        return recon_loss, kl_loss

    @torch.no_grad()
    def latent_kl_by_dim(self, x, eta):
        was_training = self.training
        self.eval()
        _, _, kl_dim_mean = self.forward(x, eta, return_kl_dim=True)
        if was_training:
            self.train()
        return kl_dim_mean

    def _sample_model_coordinates(self, eta, n_samples, eps_z=None, eps_x=None):
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        if eta.dim() == 1:
            eta = eta.unsqueeze(0)
        if eta.dim() != 2:
            raise ValueError("eta must have shape (dim_eta,) or (N, dim_eta).")

        model_parameter = next(self.parameters())
        eta = eta.to(device=model_parameter.device, dtype=model_parameter.dtype)
        if eta.shape[0] == 1:
            eta = eta.expand(n_samples, -1)
        elif eta.shape[0] != n_samples:
            raise ValueError("eta must have one row or exactly n_samples rows.")

        mu_p, lv_p = self.prior(eta)
        if eps_z is None:
            eps_z = torch.randn_like(mu_p)
        else:
            eps_z = eps_z.to(device=mu_p.device, dtype=mu_p.dtype)
            if eps_z.shape != mu_p.shape:
                raise ValueError(f"eps_z must have shape {tuple(mu_p.shape)}.")
        z = self.reparameterize(mu_p, lv_p, eps_z)

        mu_x, lv_x = self.decoder(z, eta)
        if eps_x is None:
            eps_x = torch.randn_like(mu_x)
        else:
            eps_x = eps_x.to(device=mu_x.device, dtype=mu_x.dtype)
            if eps_x.shape != mu_x.shape:
                raise ValueError(f"eps_x must have shape {tuple(mu_x.shape)}.")
        return self.reparameterize(mu_x, lv_x, eps_x)

    def _decode_physical_samples(self, model_samples):
        if self.coordinate_system == "direct":
            return model_samples
        return model_to_physical_tensor(model_samples)

    def _assert_generated_support(self, physical_samples,
                                  tolerance=SUPPORT_TOLERANCE):
        if self.coordinate_system != "transformed":
            return
        X_T = physical_samples[:, 0]
        M_T = physical_samples[:, 1]
        finite = torch.isfinite(physical_samples).all(dim=1)
        invalid = (~finite) | (
            M_T > torch.minimum(torch.zeros_like(X_T), X_T) + tolerance
        )
        invalid_count = int(invalid.sum().item())
        if invalid_count:
            raise RuntimeError(
                "Support-aware CVAE reconstruction produced invalid samples: "
                f"{invalid_count:,}/{physical_samples.shape[0]:,}. "
                "This indicates an implementation or numerical error; samples were not "
                "rejected or resampled."
            )

    def sample(self, eta, n_samples=10000, return_transformed=False,
               eps_z=None, eps_x=None, track_grad=False):
        was_training = self.training
        self.eval()
        try:
            with torch.set_grad_enabled(bool(track_grad)):
                model_samples = self._sample_model_coordinates(
                    eta, n_samples, eps_z=eps_z, eps_x=eps_x,
                )
                physical_samples = self._decode_physical_samples(model_samples)
                self._assert_generated_support(physical_samples)
                if return_transformed:
                    return model_samples
                return physical_samples
        finally:
            if was_training:
                self.train()

    def sample_differentiable(self, eta, n_samples=10000,
                              return_transformed=False,
                              eps_z=None, eps_x=None):
        return self.sample(
            eta, n_samples=n_samples,
            return_transformed=return_transformed,
            eps_z=eps_z, eps_x=eps_x, track_grad=True,
        )

    @torch.no_grad()
    def diagnose_samples(self, eta, n_samples=10000,
                         tolerance=SUPPORT_TOLERANCE, print_output=True,
                         eps_z=None, eps_x=None):
        model_samples = self.sample(
            eta, n_samples=n_samples, return_transformed=True,
            eps_z=eps_z, eps_x=eps_x,
        )
        physical_samples = self._decode_physical_samples(model_samples)
        transformed = model_samples if self.coordinate_system == "transformed" else None
        diagnostics = diagnose_cvae_samples(
            physical_samples, transformed_samples=transformed,
            tolerance=tolerance, print_output=print_output,
        )
        if self.coordinate_system == "transformed" and diagnostics["support_invalid_count"]:
            raise RuntimeError(
                "The support-aware CVAE produced a nonzero support-invalid count."
            )
        self.last_sampling_diagnostics = diagnostics
        return diagnostics

    @torch.no_grad()
    def sample_valid(self, eta, n_samples=10000, max_resample_rounds=100,
                     valid_resample=True, initial_samples=None):
        # Compatibility API: validation is diagnostic-only. No rows are rejected
        # and no replacement samples are generated in the UT model.
        del max_resample_rounds
        if initial_samples is None:
            samples = self.sample(eta, n_samples=n_samples)
        else:
            samples = initial_samples
            n_samples = int(samples.shape[0])
        diagnostics_full = diagnose_cvae_samples(
            samples, tolerance=SUPPORT_TOLERANCE, print_output=False,
        )
        if self.coordinate_system == "transformed" and diagnostics_full["support_invalid_count"]:
            raise RuntimeError(
                "The support-aware CVAE produced invalid samples; no rejection or "
                "resampling was performed."
            )
        diagnostics = {
            "validation_applied": True,
            "valid_resample": False,
            "valid_resample_requested": bool(valid_resample),
            "invalid_path_fraction": diagnostics_full["support_invalid_fraction"],
            "rejected_path_count": 0,
            "total_drawn_samples": n_samples,
            "accepted_path_count": n_samples,
            "requested_sample_count": n_samples,
            "used_sample_count": n_samples,
            "discarded_sample_count": 0,
            "discarded_sample_fraction": 0.0,
            "discarded_sample_pct": 0.0,
            "regenerated_sample_count": 0,
            "regenerated_sample_fraction": 0.0,
            "regenerated_sample_pct": 0.0,
            "support_invalid_count": diagnostics_full["support_invalid_count"],
        }
        self.last_sampling_diagnostics = diagnostics
        return samples, diagnostics

    @torch.no_grad()
    def _samples_for_pricing(self, eta, n_samples, validate_samples=True,
                             valid_resample=True, initial_samples=None):
        if initial_samples is None:
            samples = self.sample(eta, n_samples=n_samples)
        else:
            samples = initial_samples
        if validate_samples:
            return self.sample_valid(
                eta, n_samples=samples.shape[0],
                valid_resample=valid_resample,
                initial_samples=samples,
            )
        diagnostics = {
            "validation_applied": False,
            "valid_resample": False,
            "valid_resample_requested": bool(valid_resample),
            "invalid_path_fraction": None,
            "rejected_path_count": 0,
            "total_drawn_samples": int(samples.shape[0]),
            "accepted_path_count": int(samples.shape[0]),
            "requested_sample_count": int(samples.shape[0]),
            "used_sample_count": int(samples.shape[0]),
            "discarded_sample_count": 0,
            "discarded_sample_fraction": 0.0,
            "discarded_sample_pct": 0.0,
            "regenerated_sample_count": 0,
            "regenerated_sample_fraction": 0.0,
            "regenerated_sample_pct": 0.0,
            "support_invalid_count": None,
        }
        self.last_sampling_diagnostics = diagnostics
        return samples, diagnostics

    @torch.no_grad()
    def _apply_mt_correction(self, X_T, M_T, mt_corr=False):
        if mt_corr:
            upper = torch.minimum(torch.zeros_like(X_T), X_T)
            corrected_mask = M_T > upper
            M_T = torch.minimum(M_T, upper)
        else:
            corrected_mask = torch.zeros_like(M_T, dtype=torch.bool)
        corrected_count = int(corrected_mask.sum().item())
        diagnostics = {
            "mt_corr_applied": bool(mt_corr),
            "mt_corrected_sample_count": corrected_count,
            "mt_corrected_sample_fraction": corrected_count / int(M_T.shape[0]),
            "mt_corrected_sample_pct": 100.0 * corrected_count / int(M_T.shape[0]),
        }
        return M_T, diagnostics

    @torch.no_grad()
    def price_vanilla(self, eta, K, r, T, opt_type="call",
                      n_samples=10000, validate_samples=True,
                      valid_resample=True):
        del validate_samples, valid_resample
        samples = self.sample(eta, n_samples=n_samples)
        S_T = torch.exp(samples[:, 0])
        if opt_type == "call":
            payoff = torch.clamp(S_T - K, min=0.0)
        elif opt_type == "put":
            payoff = torch.clamp(K - S_T, min=0.0)
        else:
            raise ValueError("opt_type must be 'call' or 'put'.")
        return np.exp(-r * T) * payoff.mean().item()

    @torch.no_grad()
    def price_barrier(self, eta, B, K, r, T, opt_type="call",
                      n_samples=10000, validate_samples=True,
                      valid_resample=True, mt_corr=False):
        samples, diagnostics = self._samples_for_pricing(
            eta, n_samples, validate_samples=validate_samples,
            valid_resample=valid_resample,
        )
        X_T = samples[:, 0]
        M_T, correction_diagnostics = self._apply_mt_correction(
            X_T, samples[:, 1], mt_corr=mt_corr,
        )
        diagnostics.update(correction_diagnostics)
        self.last_sampling_diagnostics = diagnostics
        S_T = torch.exp(X_T)
        if not torch.isfinite(S_T).all():
            raise FloatingPointError("A CVAE path overflowed during exp(X_T).")
        alive = (M_T > np.log(B)).to(dtype=S_T.dtype)
        if opt_type == "call":
            payoff = torch.clamp(S_T - K, min=0.0) * alive
        elif opt_type == "put":
            payoff = torch.clamp(K - S_T, min=0.0) * alive
        else:
            raise ValueError("opt_type must be 'call' or 'put'.")
        return np.exp(-r * T) * payoff.mean().item()

    @torch.no_grad()
    def total_pricing(self, eta, B, K, r, T, n_samples=10000,
                      validate_samples=True, valid_resample=True,
                      mt_corr=False):
        # Exactly one generated population is shared by all four payoffs.
        samples = self.sample(eta, n_samples=n_samples)
        samples, diagnostics = self._samples_for_pricing(
            eta, n_samples, validate_samples=validate_samples,
            valid_resample=valid_resample, initial_samples=samples,
        )
        X_T = samples[:, 0]
        M_T, correction_diagnostics = self._apply_mt_correction(
            X_T, samples[:, 1], mt_corr=mt_corr,
        )
        diagnostics.update(correction_diagnostics)
        self.last_sampling_diagnostics = diagnostics

        S_T = torch.exp(X_T)
        if not torch.isfinite(S_T).all():
            raise FloatingPointError("A CVAE path overflowed during exp(X_T).")
        vanilla_call = torch.clamp(S_T - K, min=0.0)
        vanilla_put = torch.clamp(K - S_T, min=0.0)
        alive = (M_T > np.log(B)).to(dtype=S_T.dtype)
        discount = float(np.exp(-r * T))
        return {
            "van_call": discount * vanilla_call.mean().item(),
            "van_put": discount * vanilla_put.mean().item(),
            "barr_call": discount * (vanilla_call * alive).mean().item(),
            "barr_put": discount * (vanilla_put * alive).mean().item(),
            "vanilla_sample_count": int(samples.shape[0]),
            **diagnostics,
        }


# ──────────────────────────────────────
# Weighted-reconstruction UT CVAE variant
# ──────────────────────────────────────
def _normalize_weight(weight, eps=1e-8):
    return weight / (weight.mean().detach() + eps)


class CVAEBarrWeight(CVAE):
    def __init__(self,
                 dim_x=2,
                 dim_eta=7,
                 dim_z=8,
                 hidden_dims=None,
                 use_bn=False,
                 residual_blocks=0,
                 support_eps=SUPPORT_EPS,
                 coordinate_system="transformed",
                 weight_mode="barrier_put",
                 weight_alpha=3.0,
                 weight_mode2=None,
                 weight_alpha2=0.0,
                 weight_h=0.05,
                 weight_normalize=True,
                 S0=1.0,
                 K=1.0,
                 B=0.8,
                 cvae_type="barr_weight",
                 ):
        super().__init__(
            dim_x=dim_x,
            dim_eta=dim_eta,
            dim_z=dim_z,
            hidden_dims=hidden_dims,
            use_bn=use_bn,
            residual_blocks=residual_blocks,
            support_eps=support_eps,
            coordinate_system=coordinate_system,
        )
        self.weight_mode = weight_mode
        self.weight_alpha = float(weight_alpha)
        self.weight_mode2 = weight_mode2
        self.weight_alpha2 = float(weight_alpha2)
        self.weight_h = float(weight_h)
        self.weight_normalize = bool(weight_normalize)
        self.S0 = float(S0)
        self.K = float(K)
        self.B = float(B)
        self.cvae_type = cvae_type

    def checkpoint_metadata(self):
        metadata = super().checkpoint_metadata()
        metadata.update({
            "cvae_type": self.cvae_type,
            "weight_mode": self.weight_mode,
            "weight_alpha": self.weight_alpha,
            "weight_mode2": self.weight_mode2,
            "weight_alpha2": self.weight_alpha2,
            "weight_h": self.weight_h,
            "weight_normalize": self.weight_normalize,
            "S0": self.S0,
            "K": self.K,
            "B": self.B,
        })
        return metadata

    def reconstruction_weight(self, x_physical):
        mode = self.weight_mode
        mode2 = self.weight_mode2
        if mode2 in ("", "none", "None"):
            mode2 = None
        cvae_type = self.cvae_type
        valid_types = (
            "barr_weight", "barrweight", "normal_weight", "normalweight",
        )
        if cvae_type not in valid_types:
            raise ValueError(
                "reconstruction_weight supports cvae_type 'barr_weight' or "
                "'normal_weight'."
            )
        if mode not in (
            "barrier_put", "barrierput", "barrier_near", "barriernear", "all_put",
        ):
            raise ValueError("Unsupported weight_mode.")

        X_T = x_physical[:, 0]
        M_T = x_physical[:, 1]
        put_side = (X_T < float(np.log(self.K / self.S0))).to(x_physical.dtype)
        barrier_log = float(np.log(self.B / self.S0))

        if cvae_type in ("barr_weight", "barrweight"):
            near_barrier = torch.exp(-torch.abs(M_T - barrier_log) / self.weight_h)
            if mode in ("barrier_put", "barrierput"):
                weight = 1.0 + self.weight_alpha * put_side * near_barrier
            elif mode in ("barrier_near", "barriernear"):
                weight = 1.0 + self.weight_alpha * near_barrier
            else:
                raise ValueError("barr_weight requires barrier_put or barrier_near.")
        else:
            def component(component_mode, component_alpha):
                if component_mode in ("barrier_put", "barrierput"):
                    in_barrier = (M_T > barrier_log).to(x_physical.dtype)
                    return component_alpha * put_side * in_barrier
                if component_mode == "all_put":
                    return component_alpha * put_side
                raise ValueError("normal_weight requires barrier_put or all_put.")

            weight = 1.0 + component(mode, self.weight_alpha)
            if mode2 is not None:
                weight = weight + component(mode2, self.weight_alpha2)

        weight = weight.detach()
        if self.weight_normalize:
            weight = _normalize_weight(weight)
        return weight

    def additive_reconstruction_loss(self, nll_per_sample, x_physical):
        X_T = x_physical[:, 0]
        M_T = x_physical[:, 1]
        put_mask = X_T < float(np.log(self.K / self.S0))
        if self.weight_mode in ("barrier_put", "barrierput"):
            region_mask = put_mask & (M_T > float(np.log(self.B / self.S0)))
        elif self.weight_mode == "all_put":
            region_mask = put_mask
        else:
            raise ValueError("add_put_loss requires barrier_put or all_put.")

        recon_all = nll_per_sample.mean()
        mask = region_mask.to(nll_per_sample.dtype)
        recon_region = (nll_per_sample * mask).sum() / (mask.sum() + 1e-8)
        recon_loss = recon_all + self.weight_alpha * recon_region
        stats = {
            "recon_all": recon_all.detach(),
            "recon_region": recon_region.detach(),
            "region_count": region_mask.sum().detach(),
            "region_ratio": mask.mean().detach(),
        }
        return recon_loss, stats

    def forward(self, x, eta, return_kl_dim=False, return_weight_stats=False):
        # Weight masks use the physical raw [X_T, M_T]. The encoder and Gaussian
        # reconstruction NLL use the transformed [X_T, U_T].
        x_physical = x
        _, nll_per_sample, kl_dim_batch = self._elbo_components(x_physical, eta)
        if self.cvae_type in ("add_put_loss", "additive_put_loss"):
            recon_loss, weight_stats = self.additive_reconstruction_loss(
                nll_per_sample, x_physical,
            )
        else:
            weight = self.reconstruction_weight(x_physical)
            recon_loss = (weight * nll_per_sample).mean()
            weight_stats = {
                "mean": weight.mean().detach(),
                "min": weight.min().detach(),
                "max": weight.max().detach(),
            }

        kl_dim_mean = kl_dim_batch.mean(dim=0)
        kl_loss = kl_dim_mean.sum()
        if return_weight_stats:
            if return_kl_dim:
                return recon_loss, kl_loss, kl_dim_mean, weight_stats
            return recon_loss, kl_loss, weight_stats
        if return_kl_dim:
            return recon_loss, kl_loss, kl_dim_mean
        return recon_loss, kl_loss
