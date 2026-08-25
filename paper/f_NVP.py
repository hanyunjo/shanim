import glob
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn

LOG_2PI = math.log(2.0 * math.pi)
SUPPORT_EPS = 1e-6
SUPPORT_TOLERANCE = 1e-6
UT_MODEL_COORDINATES = ["X_T", "U_T"]
MT_MODEL_COORDINATES = ["X_T", "M_T"]
PHYSICAL_COORDINATES = ["X_T", "M_T"]
SUPPORT_PARAMETERIZATION = "M_T = min(0, X_T) - softplus(U_T)"


def _canonical_target_parameterization(target_parameterization):
    aliases = {
        None: "mt",
        "mt": "mt",
        "direct": "mt",
        "legacy": "mt",
        "ut": "ut",
        "transformed": "ut",
        "support_aware": "ut",
    }
    key = (
        None if target_parameterization is None
        else str(target_parameterization).strip().lower()
    )
    if key not in aliases:
        raise ValueError("target_parameterization must be 'mt' or 'ut'.")
    return aliases[key]


def _checkpoint_target_parameterization(checkpoint):
    saved = checkpoint.get("target_parameterization")
    if saved is not None:
        return _canonical_target_parameterization(saved)
    coordinates = checkpoint.get("model_coordinates")
    if coordinates is None or list(coordinates) == MT_MODEL_COORDINATES:
        return "mt"
    if list(coordinates) == UT_MODEL_COORDINATES:
        return "ut"
    raise ValueError(f"Unsupported checkpoint model_coordinates: {coordinates!r}")


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


def running_min_to_u_tensor(X_T, M_T, eps=SUPPORT_EPS,
                            tolerance=SUPPORT_TOLERANCE):
    if X_T.shape != M_T.shape:
        raise ValueError("X_T and M_T must have the same shape.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    finite = torch.isfinite(X_T) & torch.isfinite(M_T)
    upper = torch.minimum(torch.zeros_like(X_T), X_T)
    invalid = (~finite) | (M_T > upper + tolerance)
    invalid_count = int(invalid.sum().item())
    if invalid_count:
        raise ValueError(
            "Raw NVP training data violates M_T <= min(0, X_T) + tolerance: "
            f"invalid={invalid_count:,}/{X_T.numel():,}. "
            "The samples were not corrected or dropped."
        )
    D_T = upper - M_T
    return inverse_softplus_tensor(torch.clamp(D_T, min=eps))


def mt_to_ut_tensor(x_physical, eps=SUPPORT_EPS,
                    tolerance=SUPPORT_TOLERANCE):
    if x_physical.ndim != 2 or x_physical.shape[1] != 2:
        raise ValueError("Expected [X_T, M_T] with shape (N, 2).")
    X_T = x_physical[:, 0]
    U_T = running_min_to_u_tensor(
        X_T, x_physical[:, 1], eps=eps, tolerance=tolerance,
    )
    return torch.stack((X_T, U_T), dim=1)


def ut_to_mt_tensor(x_model):
    if x_model.ndim != 2 or x_model.shape[1] != 2:
        raise ValueError("Expected [X_T, U_T] with shape (N, 2).")
    X_T = x_model[:, 0]
    U_T = x_model[:, 1]
    M_T = torch.minimum(torch.zeros_like(X_T), X_T) - torch.nn.functional.softplus(U_T)
    return torch.stack((X_T, M_T), dim=1)


def _support_invalid_mask(samples, tolerance=SUPPORT_TOLERANCE):
    X_T = samples[:, 0]
    M_T = samples[:, 1]
    finite = torch.isfinite(samples).all(dim=1)
    return (~finite) | (
        M_T > torch.minimum(torch.zeros_like(X_T), X_T) + tolerance
    )

# The paper defines M_T as a running maximum. In this repository, however,
# paths[:, 2] is produced by X.min(axis=1), so the trained pair is
# (X_T, running minimum). The CRealNVP architecture still applies, but the
# target and barrier payoff must be interpreted accordingly.
DATASET_PRESETS = {
    "bs": {
        "chunk_dirs": ("/mnt/d/bs_chunks_correction",),
        "eta_paths": ("/mnt/d/bs_eta_basic.h5",),
        "dim_eta": 3,
    },
    "hes": {
        "chunk_dirs": ("/mnt/d/heston_chunks_correction",),
        "eta_paths": ("/mnt/d/heston_eta_basic.h5",),
        "dim_eta": 7,
    },
    "hes_clip": {
        "chunk_dirs": (
            "/mnt/d/heston_clip_x2_chunks_correction",
            "/mnt/e/heston_clip_x2_chunks_correction",
        ),
        "eta_paths": (
            "/mnt/d/heston_eta_clip_x2.h5",
            "/mnt/e/heston_eta_clip_x2.h5",
        ),
        "dim_eta": 7,
    },
}


def _canonical_model_type(model_type: str) -> str:
    aliases = {
        "bs": "bs",
        "black_scholes": "bs",
        "hes": "hes",
        "heston": "hes",
        "hes_clip": "hes_clip",
        "heston_clip": "hes_clip",
        "expou": "expou",
    }
    key = str(model_type).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown model_type={model_type!r}; expected one of {sorted(aliases)}")
    return aliases[key]


def _canonical_target_pair(target_pair: str) -> str:
    key = str(target_pair).upper().replace(" ", "").replace(",", "_")
    aliases = {
        "XT_MT": "XT_MIN",
        "XT_MIN": "XT_MIN",
        "XT_MINIMUM": "XT_MIN",
    }
    if key not in aliases:
        raise ValueError(
            "This repository's HDF5 chunks only contain [eta_index, X_T, running_minimum]. "
            "Use target_pair='XT_MIN' (legacy alias 'XT_MT' is also accepted)."
        )
    return aliases[key]


def _first_existing(paths) -> Optional[str]:
    return next((p for p in paths if os.path.exists(p)), None)


def resolve_dataset_paths(model_type: str, chunk_dir=None, eta_path=None):
    """Resolve the repository's current HDF5 locations, with D:/E: fallback."""
    model_type = _canonical_model_type(model_type)
    preset = DATASET_PRESETS.get(model_type)
    if chunk_dir is None:
        if preset is None:
            raise ValueError(f"chunk_dir is required for model_type={model_type!r}")
        chunk_dir = _first_existing(preset["chunk_dirs"])
    if eta_path is None:
        if preset is None:
            raise ValueError(f"eta_path is required for model_type={model_type!r}")
        eta_path = _first_existing(preset["eta_paths"])
    if chunk_dir is None or not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
    if eta_path is None or not os.path.isfile(eta_path):
        raise FileNotFoundError(f"Eta file not found: {eta_path}")
    return os.fspath(chunk_dir), os.fspath(eta_path)


def _chunk_index(path: str) -> int:
    m = re.search(r"chunk_(\d+)", os.path.basename(path))
    return -1 if m is None else int(m.group(1))


def list_chunk_paths(chunk_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(chunk_dir, "*.h5"))
    malformed = [p for p in paths if _chunk_index(p) < 0]
    if malformed:
        raise ValueError(f"Cannot parse chunk index from: {malformed}")
    paths = sorted(paths, key=lambda p: (_chunk_index(p), p))
    indices = [_chunk_index(p) for p in paths]
    duplicates = sorted({i for i in indices if indices.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate chunk indices in {chunk_dir}: {duplicates}")
    return paths


def load_eta_stats(eta_path: str, expected_dim: Optional[int] = None):
    with h5py.File(eta_path, "r") as f:
        if "etas" not in f:
            raise KeyError(f"{eta_path} does not contain an 'etas' dataset")
        etas = f["etas"][:].astype(np.float32)
    if etas.ndim != 2 or etas.shape[0] == 0:
        raise ValueError(f"Expected etas with shape (N, dim_eta), got {etas.shape}")
    if expected_dim is not None and etas.shape[1] != expected_dim:
        raise ValueError(f"Expected dim_eta={expected_dim}, got {etas.shape[1]} in {eta_path}")
    eta_min = etas.min(axis=0)
    eta_max = etas.max(axis=0)
    if not (np.isfinite(eta_min).all() and np.isfinite(eta_max).all()):
        raise ValueError(f"Non-finite eta values found in {eta_path}")
    return etas, eta_min, eta_max


def normalize_eta_np(eta, eta_min, eta_max):
    span = eta_max - eta_min
    safe_span = np.where(span > 0, span, 1.0)
    normalized = (eta - eta_min) / safe_span
    if np.any(span == 0):
        normalized[..., span == 0] = 0.0
    return normalized


class ScaleMLP(nn.Module):
    """2022 paper s-net: 4 hidden layers, 100 nodes, hidden activation 3*tanh."""
    def __init__(self, in_dim, hidden_dim=100, n_hidden=4, use_bn=True):
        super().__init__()
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        d = in_dim
        for _ in range(n_hidden):
            self.linears.append(nn.Linear(d, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
            d = hidden_dim
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        h = x
        for linear, bn in zip(self.linears, self.bns):
            h = 3.0 * torch.tanh(bn(linear(h)))
        return self.out(h)


class TranslationMLP(nn.Module):
    """2022 paper t-net: 4 hidden layers, 100 nodes, LeakyReLU; Heston slope=0.5."""
    def __init__(self, in_dim, hidden_dim=100, n_hidden=4, negative_slope=0.5, use_bn=True):
        super().__init__()
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.negative_slope = float(negative_slope)
        d = in_dim
        for _ in range(n_hidden):
            self.linears.append(nn.Linear(d, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
            d = hidden_dim
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        h = x
        for linear, bn in zip(self.linears, self.bns):
            h = torch.nn.functional.leaky_relu(bn(linear(h)), negative_slope=self.negative_slope)
        return self.out(h)


class ConditionalAffineCoupling2D(nn.Module):
    def __init__(self, dim_eta, fixed_idx, hidden_dim=100, n_hidden=4,
                 t_negative_slope=0.5, use_bn=True, scale_clip=None):
        super().__init__()
        self.fixed_idx = int(fixed_idx)
        self.trans_idx = 1 - self.fixed_idx
        self.scale_clip = scale_clip
        net_in = 1 + dim_eta
        self.s_net = ScaleMLP(net_in, hidden_dim, n_hidden, use_bn)
        self.t_net = TranslationMLP(net_in, hidden_dim, n_hidden, t_negative_slope, use_bn)

    def _st(self, fixed, eta):
        cond = torch.cat([fixed, eta], dim=1)
        s = self.s_net(cond)
        t = self.t_net(cond)
        if self.scale_clip is not None:
            s = torch.clamp(s, -self.scale_clip, self.scale_clip)
        return s, t

    def forward(self, x, eta):
        fixed = x[:, self.fixed_idx:self.fixed_idx+1]
        transformed = x[:, self.trans_idx:self.trans_idx+1]
        s, t = self._st(fixed, eta)
        y = x.clone()
        y[:, self.trans_idx:self.trans_idx+1] = transformed * torch.exp(s) + t
        return y, s.squeeze(1)

    def inverse(self, y, eta):
        fixed = y[:, self.fixed_idx:self.fixed_idx+1]
        transformed = y[:, self.trans_idx:self.trans_idx+1]
        s, t = self._st(fixed, eta)
        x = y.clone()
        x[:, self.trans_idx:self.trans_idx+1] = (transformed - t) * torch.exp(-s)
        return x


class CRealNVP2D(nn.Module):
    def __init__(self, dim_eta, n_coupling=6, hidden_dim=100, n_hidden=4,
                 t_negative_slope=0.5, use_bn=True, scale_clip=None,
                 target_parameterization="mt", support_eps=SUPPORT_EPS):
        super().__init__()
        self.dim_x = 2
        self.dim_eta = dim_eta
        self.n_coupling = n_coupling
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.t_negative_slope = t_negative_slope
        self.use_bn = use_bn
        self.scale_clip = scale_clip
        self.target_parameterization = _canonical_target_parameterization(
            target_parameterization
        )
        self.support_eps = float(support_eps)
        if self.support_eps <= 0:
            raise ValueError("support_eps must be positive.")
        self.model_coordinates = (
            list(UT_MODEL_COORDINATES)
            if self.target_parameterization == "ut"
            else list(MT_MODEL_COORDINATES)
        )
        self.physical_coordinates = list(PHYSICAL_COORDINATES)
        self.bn_frozen = False
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling2D(
                dim_eta=dim_eta,
                fixed_idx=i % 2,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                t_negative_slope=t_negative_slope,
                use_bn=use_bn,
                scale_clip=scale_clip,
            ) for i in range(n_coupling)
        ])

    def checkpoint_metadata(self):
        return {
            "target_parameterization": self.target_parameterization,
            "model_coordinates": list(self.model_coordinates),
            "physical_coordinates": list(self.physical_coordinates),
            "support_parameterization": (
                SUPPORT_PARAMETERIZATION
                if self.target_parameterization == "ut"
                else "model output M_T"
            ),
            "support_eps": self.support_eps,
            "path_statistic": "running_minimum",
        }

    def prepare_training_target(self, x_physical):
        if self.target_parameterization == "mt":
            return x_physical
        return mt_to_ut_tensor(x_physical, eps=self.support_eps)

    def model_to_physical(self, x_model):
        if self.target_parameterization == "mt":
            return x_model
        return ut_to_mt_tensor(x_model)

    def _assert_generated_support(self, physical_samples):
        if self.target_parameterization != "ut":
            return
        invalid_count = int(_support_invalid_mask(physical_samples).sum().item())
        if invalid_count:
            raise RuntimeError(
                "Support-aware NVP reconstruction produced invalid samples: "
                f"{invalid_count:,}/{physical_samples.shape[0]:,}. "
                "No samples were rejected or resampled."
            )

    def encode(self, x_model, eta):
        z = x_model
        log_det = torch.zeros(
            x_model.shape[0], device=x_model.device, dtype=x_model.dtype,
        )
        for layer in self.layers:
            z, ld = layer(z, eta)
            log_det += ld
        return z, log_det

    def decode(self, z, eta):
        x_model = z
        for layer in reversed(self.layers):
            x_model = layer.inverse(x_model, eta)
        return x_model

    def decode_physical(self, z, eta):
        x_model = self.decode(z, eta)
        physical_samples = self.model_to_physical(x_model)
        self._assert_generated_support(physical_samples)
        return physical_samples

    def log_prob(self, x_physical, eta):
        x_model = self.prepare_training_target(x_physical)
        z, log_det = self.encode(x_model, eta)
        log_pz = -0.5 * (z.pow(2) + LOG_2PI).sum(dim=1)
        return log_pz + log_det

    def nll(self, x_physical, eta):
        return -self.log_prob(x_physical, eta).mean()

    @torch.no_grad()
    def sample(self, eta, n_samples, antithetic=False, generator=None,
               return_transformed=False):
        self.eval()
        if eta.ndim == 1:
            eta = eta.unsqueeze(0)
        if eta.shape[0] == 1:
            eta = eta.expand(n_samples, -1)
        elif eta.shape[0] != n_samples:
            raise ValueError("eta must have one row or n_samples rows")

        if antithetic:
            half = (n_samples + 1) // 2
            z0 = torch.randn(
                half, 2, device=eta.device, dtype=eta.dtype, generator=generator,
            )
            z = torch.cat([z0, -z0], dim=0)[:n_samples]
        else:
            z = torch.randn(
                n_samples, 2, device=eta.device, dtype=eta.dtype,
                generator=generator,
            )
        model_samples = self.decode(z, eta)
        physical_samples = self.model_to_physical(model_samples)
        self._assert_generated_support(physical_samples)
        if return_transformed:
            return model_samples
        return physical_samples


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            if module.affine:
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
    if isinstance(model, CRealNVP2D):
        model.bn_frozen = True


def enforce_frozen_batchnorm(model):
    if isinstance(model, CRealNVP2D) and model.bn_frozen:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()


def _prepare_paths_batch(paths, etas, eta_min, eta_max, device, source, validate=True):
    if paths.ndim != 2 or paths.shape[1] != 3:
        raise ValueError(f"{source}: expected paths shape (N, 3), got {paths.shape}")

    raw_idx = paths[:, 0]
    rounded_idx = np.rint(raw_idx)
    if validate and (not np.isfinite(raw_idx).all() or not np.array_equal(raw_idx, rounded_idx)):
        raise ValueError(f"{source}: eta indices must be finite integer-valued numbers")
    ori_idx = rounded_idx.astype(np.int64, copy=False)
    if ori_idx.size and (ori_idx.min() < 0 or ori_idx.max() >= len(etas)):
        raise IndexError(
            f"{source}: eta index range [{ori_idx.min()}, {ori_idx.max()}] "
            f"is outside [0, {len(etas) - 1}]"
        )

    x_np = paths[:, 1:3].astype(np.float32, copy=True)
    if validate:
        if not np.isfinite(x_np).all():
            raise ValueError(f"{source}: non-finite X_T/running-minimum values found")
        # X_0 = 0 is included by the generator, hence min_t X_t <= min(0, X_T).
        invalid_min = x_np[:, 1] > np.minimum(0.0, x_np[:, 0]) + 1e-6
        if invalid_min.any():
            raise ValueError(
                f"{source}: paths[:, 2] is not a running minimum for "
                f"{int(invalid_min.sum())} rows"
            )

    eta_np = etas[ori_idx].astype(np.float32, copy=True)
    eta_np = normalize_eta_np(eta_np, eta_min, eta_max).astype(np.float32, copy=False)
    if validate and not np.isfinite(eta_np).all():
        raise ValueError(f"{source}: non-finite normalized eta values found")
    return torch.from_numpy(x_np).to(device), torch.from_numpy(eta_np).to(device)


def iter_chunk_batches(
    chunk_path, etas, eta_min, eta_max, batch_size, device,
    *, drop_last=False, validate=True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Stream a 67M-row HDF5 chunk instead of materializing it on GPU."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    with h5py.File(chunk_path, "r") as f:
        if "paths" not in f:
            raise KeyError(f"{chunk_path} does not contain a 'paths' dataset")
        paths_ds = f["paths"]
        if paths_ds.ndim != 2 or paths_ds.shape[1] != 3:
            raise ValueError(f"{chunk_path}: expected paths shape (N, 3), got {paths_ds.shape}")
        n_rows = int(paths_ds.shape[0])
        stop = (n_rows // batch_size) * batch_size if drop_last else n_rows
        for start in range(0, stop, batch_size):
            end = min(start + batch_size, stop)
            source = f"{chunk_path}[{start}:{end}]"
            yield _prepare_paths_batch(
                paths_ds[start:end], etas, eta_min, eta_max, device, source, validate
            )


def load_chunk_to_device(
    chunk_path, etas, eta_min, eta_max, device, validate=True,
    reserve_bytes=1024**3,
):
    """Load one complete HDF5 chunk as float32 x/eta tensors on a CUDA device."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Full-chunk loading requires a CUDA device")

    load_start = time.perf_counter()
    with h5py.File(chunk_path, "r") as f:
        if "paths" not in f:
            raise KeyError(f"{chunk_path} does not contain a 'paths' dataset")
        paths_ds = f["paths"]
        if paths_ds.ndim != 2 or paths_ds.shape[1] != 3:
            raise ValueError(f"{chunk_path}: expected paths shape (N, 3), got {paths_ds.shape}")

        n_rows = int(paths_ds.shape[0])
        required_bytes = n_rows * (2 + etas.shape[1]) * np.dtype(np.float32).itemsize
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if required_bytes + reserve_bytes > free_bytes:
            raise MemoryError(
                f"{chunk_path}: full GPU chunk needs about {required_bytes / 1024**3:.2f} GiB "
                f"plus {reserve_bytes / 1024**3:.2f} GiB reserved for model activations, "
                f"but only {free_bytes / 1024**3:.2f} GiB is free. "
                "Use memory_on_gpu=False or reduce other GPU memory usage."
            )

        raw_idx = paths_ds[:, 0]
        rounded_idx = np.rint(raw_idx)
        if validate and (
            not np.isfinite(raw_idx).all()
            or not np.array_equal(raw_idx, rounded_idx)
        ):
            raise ValueError(f"{chunk_path}: eta indices must be finite integer-valued numbers")
        ori_idx = rounded_idx.astype(np.int64, copy=False)
        del raw_idx, rounded_idx

        if ori_idx.size and (ori_idx.min() < 0 or ori_idx.max() >= len(etas)):
            raise IndexError(
                f"{chunk_path}: eta index range [{ori_idx.min()}, {ori_idx.max()}] "
                f"is outside [0, {len(etas) - 1}]"
            )

        x_np = paths_ds[:, 1:3].astype(np.float32, copy=True)

    if validate:
        if not np.isfinite(x_np).all():
            raise ValueError(f"{chunk_path}: non-finite X_T/running-minimum values found")
        invalid_min = x_np[:, 1] > np.minimum(0.0, x_np[:, 0]) + 1e-6
        if invalid_min.any():
            raise ValueError(
                f"{chunk_path}: paths[:, 2] is not a running minimum for "
                f"{int(invalid_min.sum())} rows"
            )
        del invalid_min

    x_chunk = torch.from_numpy(x_np).to(device)
    del x_np

    eta_np = etas[ori_idx].astype(np.float32, copy=True)
    del ori_idx
    span = eta_max - eta_min
    safe_span = np.where(span > 0, span, 1.0).astype(np.float32, copy=False)
    eta_np -= eta_min
    eta_np /= safe_span
    if np.any(span == 0):
        eta_np[:, span == 0] = 0.0
    if validate and not np.isfinite(eta_np).all():
        raise ValueError(f"{chunk_path}: non-finite normalized eta values found")

    eta_chunk = torch.from_numpy(eta_np).to(device)
    del eta_np

    elapsed = time.perf_counter() - load_start
    data_gib = required_bytes / 1024**3
    print(
        f"[GPU chunk loaded] {os.path.basename(chunk_path)} | "
        f"rows={n_rows:,} | data={data_gib:.2f} GiB | time={elapsed:.1f}s",
        flush=True,
    )
    return x_chunk, eta_chunk


def iter_device_chunk_batches(
    chunk_path, etas, eta_min, eta_max, batch_size, device,
    *, drop_last=False, validate=True,
):
    """Load one complete chunk on GPU, then yield GPU tensor slices."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    x_chunk, eta_chunk = load_chunk_to_device(
        chunk_path,
        etas,
        eta_min,
        eta_max,
        device,
        validate=validate,
    )
    n_rows = x_chunk.shape[0]
    stop = (n_rows // batch_size) * batch_size if drop_last else n_rows

    try:
        for start in range(0, stop, batch_size):
            end = min(start + batch_size, stop)
            yield x_chunk[start:end], eta_chunk[start:end]
    finally:
        del x_chunk, eta_chunk


def chunk_row_counts(paths):
    out = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            if "paths" not in f:
                raise KeyError(f"{path} does not contain a 'paths' dataset")
            dataset = f["paths"]
            if dataset.ndim != 2 or dataset.shape[1] != 3:
                raise ValueError(f"{path}: expected paths shape (N, 3), got {dataset.shape}")
            out[path] = int(dataset.shape[0])
    return out


def paper2022_lr_schedule(model_type="hes", target_pair="XT_MIN"):
    model_type = _canonical_model_type(model_type)
    target_pair = _canonical_target_pair(target_pair)
    # The paper adds two 1e-6 epochs for Heston (X_T, running maximum).
    # We keep that schedule for the repository's analogous running-minimum pair.
    if model_type in {"hes", "hes_clip"} and target_pair == "XT_MIN":
        return [1e-5, 1e-5, 1e-6, 1e-6, 1e-6]
    return [1e-5, 1e-5, 1e-6]


def train_crealnvp_paper2022(
    *, save_path, chunk_dir=None, eta_path=None, model_type="hes", target_pair="XT_MIN",
    target_parameterization="mt",
    batch_size=16384, train_chunk_idxs=None, validation_chunk_idxs=None,
    shuffle_chunks=True, seed=1234, device=None, scale_clip=None,
    hidden_dim=100, n_hidden=4,
    bn_pretrain_fraction=0.05, drop_last=True, validate_data=True,
    memory_on_gpu=True, lr=None, num_epochs=None, num_chunks=None,
    val_every_chunks=None, tmp_save_every_chunks=10, resume_path=None,
):
    """
    Train the paper's CRealNVP architecture on this repository's HDF5 schema.

    With no custom learning-rate arguments, the original paper schedule is used.
    Set lr and num_epochs=1 for one notebook-controlled epoch, or set
    num_chunks to control the number of chunk files processed in this call.
    val_every_chunks controls intermediate validation without changing the
    model architecture or objective. target_parameterization="mt" learns the
    raw [X_T, M_T] pair; "ut" validates that raw pair and trains the same flow
    architecture on [X_T, U_T]. The HDF5 format is unchanged. hidden_dim is
    the width of each s/t hidden layer, while n_hidden is the number of hidden
    layers in each s/t network.
    """
    model_type = _canonical_model_type(model_type)
    target_pair = _canonical_target_pair(target_pair)
    target_parameterization = _canonical_target_parameterization(
        target_parameterization
    )
    batch_size = int(batch_size)
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 because BatchNorm is used during pretraining")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, (int, np.integer)):
        raise TypeError("hidden_dim must be a positive integer")
    hidden_dim = int(hidden_dim)
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")
    if isinstance(n_hidden, bool) or not isinstance(n_hidden, (int, np.integer)):
        raise TypeError("n_hidden must be a positive integer")
    n_hidden = int(n_hidden)
    if n_hidden < 1:
        raise ValueError("n_hidden must be >= 1")
    if not 0.0 <= bn_pretrain_fraction <= 1.0:
        raise ValueError("bn_pretrain_fraction must be between 0 and 1")

    if lr is not None:
        lr = float(lr)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be a finite positive number")

    if num_epochs is not None:
        if isinstance(num_epochs, bool) or not isinstance(num_epochs, (int, np.integer)):
            raise TypeError("num_epochs must be a positive integer")
        num_epochs = int(num_epochs)
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")

    if num_chunks is not None:
        if isinstance(num_chunks, bool) or not isinstance(num_chunks, (int, np.integer)):
            raise TypeError("num_chunks must be a positive integer")
        num_chunks = int(num_chunks)
        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        if num_epochs is not None:
            raise ValueError("Use either num_chunks or num_epochs, not both")

    if val_every_chunks is not None:
        if (
            isinstance(val_every_chunks, bool)
            or not isinstance(val_every_chunks, (int, np.integer))
        ):
            raise TypeError("val_every_chunks must be a positive integer or None")
        val_every_chunks = int(val_every_chunks)
        if val_every_chunks < 1:
            raise ValueError("val_every_chunks must be >= 1")

    if tmp_save_every_chunks is not None:
        if (
            isinstance(tmp_save_every_chunks, bool)
            or not isinstance(tmp_save_every_chunks, (int, np.integer))
        ):
            raise TypeError("tmp_save_every_chunks must be a positive integer or None")
        tmp_save_every_chunks = int(tmp_save_every_chunks)
        if tmp_save_every_chunks < 1:
            raise ValueError("tmp_save_every_chunks must be >= 1 or None")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not isinstance(memory_on_gpu, (bool, np.bool_)):
        raise TypeError("memory_on_gpu must be True or False")
    memory_on_gpu = bool(memory_on_gpu)
    if memory_on_gpu and device.type != "cuda":
        raise ValueError("memory_on_gpu=True requires a CUDA device")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    save_path = Path(save_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None
    if resume_path is not None:
        resume_path = Path(resume_path).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        checkpoint_model_type = _canonical_model_type(
            resume_checkpoint.get("model_type", model_type)
        )
        if checkpoint_model_type != model_type:
            raise ValueError(
                f"checkpoint model_type={checkpoint_model_type}, current model_type={model_type}"
            )
        checkpoint_target_pair = _canonical_target_pair(
            resume_checkpoint.get("target_pair", target_pair)
        )
        if checkpoint_target_pair != target_pair:
            raise ValueError(
                f"checkpoint target_pair={checkpoint_target_pair}, "
                f"current target_pair={target_pair}"
            )
        checkpoint_target_parameterization = _checkpoint_target_parameterization(
            resume_checkpoint
        )
        if checkpoint_target_parameterization != target_parameterization:
            raise ValueError(
                "checkpoint target_parameterization="
                f"{checkpoint_target_parameterization}, current="
                f"{target_parameterization}. MT and UT checkpoints cannot be mixed."
            )
        checkpoint_hidden_dim = int(resume_checkpoint.get("hidden_dim", 100))
        checkpoint_n_hidden = int(resume_checkpoint.get("n_hidden", 4))
        if checkpoint_hidden_dim != hidden_dim or checkpoint_n_hidden != n_hidden:
            raise ValueError(
                "NVP architecture differs from the resume checkpoint: "
                f"checkpoint hidden_dim={checkpoint_hidden_dim}, "
                f"n_hidden={checkpoint_n_hidden}; current hidden_dim={hidden_dim}, "
                f"n_hidden={n_hidden}. Start a new experiment or pass the saved values."
            )

        if chunk_dir is None:
            chunk_dir = resume_checkpoint.get("chunk_dir")
        if eta_path is None:
            eta_path = resume_checkpoint.get("eta_path")

        checkpoint_train_idxs = resume_checkpoint.get("train_chunk_idxs")
        if train_chunk_idxs is None:
            train_chunk_idxs = checkpoint_train_idxs
        elif checkpoint_train_idxs is not None and {
            int(i) for i in train_chunk_idxs
        } != {int(i) for i in checkpoint_train_idxs}:
            raise ValueError("train_chunk_idxs differs from the resume checkpoint")

        checkpoint_val_idxs = resume_checkpoint.get("validation_chunk_idxs")
        if validation_chunk_idxs is None:
            validation_chunk_idxs = checkpoint_val_idxs
        elif checkpoint_val_idxs is not None and {
            int(i) for i in validation_chunk_idxs
        } != {int(i) for i in checkpoint_val_idxs}:
            raise ValueError("validation_chunk_idxs differs from the resume checkpoint")

        checkpoint_scale_clip = resume_checkpoint.get("scale_clip")
        if scale_clip is not None and scale_clip != checkpoint_scale_clip:
            raise ValueError(
                f"checkpoint scale_clip={checkpoint_scale_clip}, current scale_clip={scale_clip}"
            )
        scale_clip = checkpoint_scale_clip

    chunk_dir, eta_path = resolve_dataset_paths(model_type, chunk_dir, eta_path)

    all_paths = list_chunk_paths(chunk_dir)
    if not all_paths:
        raise FileNotFoundError(f"No .h5 files in {chunk_dir}")
    path_by_idx = {_chunk_index(p): p for p in all_paths}
    all_idxs = sorted(path_by_idx)

    val_idxs = {int(i) for i in (validation_chunk_idxs or [])}
    missing_val = sorted(val_idxs - set(all_idxs))
    if missing_val:
        raise ValueError(f"Validation chunk indices not found: {missing_val}")

    if train_chunk_idxs is None:
        train_idxs = [i for i in all_idxs if i not in val_idxs]
    else:
        train_idxs = list(dict.fromkeys(int(i) for i in train_chunk_idxs))
        missing_train = sorted(set(train_idxs) - set(all_idxs))
        if missing_train:
            raise ValueError(f"Training chunk indices not found: {missing_train}")
        overlap = sorted(set(train_idxs) & val_idxs)
        if overlap:
            raise ValueError(f"Chunks cannot be both training and validation: {overlap}")
    if not train_idxs:
        raise ValueError("At least one training chunk is required")

    preset = DATASET_PRESETS.get(model_type)
    expected_dim = None if preset is None else preset["dim_eta"]
    etas, eta_min, eta_max = load_eta_stats(eta_path, expected_dim=expected_dim)
    dim_eta = etas.shape[1]

    selected_paths = [path_by_idx[i] for i in sorted(set(train_idxs) | val_idxs)]
    counts = chunk_row_counts(selected_paths)
    rows_per_epoch = sum(counts[path_by_idx[i]] for i in train_idxs)
    if drop_last:
        effective_rows_per_epoch = sum(
            (counts[path_by_idx[i]] // batch_size) * batch_size for i in train_idxs
        )
    else:
        effective_rows_per_epoch = rows_per_epoch
    if effective_rows_per_epoch == 0:
        raise ValueError("No complete training batches are available")
    bn_freeze_after_rows = int(math.ceil(bn_pretrain_fraction * effective_rows_per_epoch))

    paper_lr_schedule = paper2022_lr_schedule(model_type, target_pair)

    if resume_checkpoint is None:
        # Paper: Heston slope=0.5 and ExpOU slope=0.3. BS is an adaptation and uses 0.5.
        t_slope = 0.3 if model_type == "expou" else 0.5
        model = CRealNVP2D(
            dim_eta=dim_eta,
            n_coupling=6,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            t_negative_slope=t_slope,
            use_bn=True,
            scale_clip=scale_clip,
            target_parameterization=target_parameterization,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr or paper_lr_schedule[0])
        history = {"epoch_nll": [], "val_nll": [], "lr": [], "epoch_seconds": []}
        completed_epochs = 0
        epoch1_rows_seen = 0
        bn_has_been_frozen = False
    else:
        checkpoint_dim_eta = int(resume_checkpoint["dim_eta"])
        if checkpoint_dim_eta != dim_eta:
            raise ValueError(
                f"checkpoint dim_eta={checkpoint_dim_eta}, current dim_eta={dim_eta}"
            )

        t_slope = float(resume_checkpoint["t_negative_slope"])
        model = CRealNVP2D(
            dim_eta=dim_eta,
            n_coupling=int(resume_checkpoint["n_coupling"]),
            hidden_dim=int(resume_checkpoint["hidden_dim"]),
            n_hidden=int(resume_checkpoint["n_hidden"]),
            t_negative_slope=t_slope,
            use_bn=bool(resume_checkpoint["use_bn"]),
            scale_clip=scale_clip,
            target_parameterization=target_parameterization,
            support_eps=float(resume_checkpoint.get("support_eps", SUPPORT_EPS)),
        ).to(device)
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr or paper_lr_schedule[0])
        if "optimizer_state" not in resume_checkpoint:
            raise KeyError("resume checkpoint does not contain optimizer_state")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])

        saved_history = resume_checkpoint.get("history", {})
        history = {
            "epoch_nll": list(saved_history.get("epoch_nll", [])),
            "val_nll": list(saved_history.get("val_nll", [])),
            "lr": list(saved_history.get("lr", [])),
            "epoch_seconds": list(saved_history.get("epoch_seconds", [])),
        }
        history_lengths = {len(values) for values in history.values()}
        if len(history_lengths) != 1:
            raise ValueError("resume checkpoint history lists have inconsistent lengths")

        completed_epochs = int(
            resume_checkpoint.get("completed_epochs", len(history["epoch_nll"]))
        )
        if completed_epochs != len(history["epoch_nll"]):
            raise ValueError(
                f"checkpoint completed_epochs={completed_epochs}, "
                f"but history contains {len(history['epoch_nll'])} epochs"
            )

        epoch1_rows_seen = int(
            resume_checkpoint.get(
                "epoch1_rows_seen",
                effective_rows_per_epoch if completed_epochs >= 1 else 0,
            )
        )
        bn_has_been_frozen = bool(
            resume_checkpoint.get("bn_frozen", completed_epochs >= 1)
        )
        if bn_has_been_frozen:
            freeze_batchnorm(model)

    chunk_history_keys = (
        "epoch", "global_chunk", "chunk_pos", "chunk_idx", "chunk_file",
        "nll", "epoch_running_nll", "lr", "seconds", "gpu_memory_gib",
    )
    if resume_checkpoint is None:
        chunk_loss_history = {key: [] for key in chunk_history_keys}
        completed_chunks = 0
    else:
        saved_chunk_history = resume_checkpoint.get("chunk_loss_history", {})
        chunk_loss_history = {
            key: list(saved_chunk_history.get(key, []))
            for key in chunk_history_keys
        }
        chunk_history_lengths = {
            len(values) for values in chunk_loss_history.values()
        }
        if len(chunk_history_lengths) != 1:
            raise ValueError(
                "resume checkpoint chunk_loss_history lists have inconsistent lengths"
            )
        recorded_chunks = len(chunk_loss_history["nll"])
        completed_chunks = int(
            resume_checkpoint.get(
                "completed_chunks",
                recorded_chunks or completed_epochs * len(train_idxs),
            )
        )
        if recorded_chunks > completed_chunks:
            raise ValueError(
                f"checkpoint completed_chunks={completed_chunks}, "
                f"but chunk history contains {recorded_chunks} chunks"
            )

    validation_history_keys = (
        "global_chunk", "epoch", "chunk_pos", "nll", "lr", "seconds",
    )
    if resume_checkpoint is None:
        validation_history = {key: [] for key in validation_history_keys}
        epoch_accum = {"loss_sum": 0.0, "sample_count": 0, "seconds": 0.0}
    else:
        saved_validation_history = resume_checkpoint.get("validation_history", {})
        validation_history = {
            key: list(saved_validation_history.get(key, []))
            for key in validation_history_keys
        }
        validation_history_lengths = {
            len(values) for values in validation_history.values()
        }
        if len(validation_history_lengths) != 1:
            raise ValueError(
                "resume checkpoint validation_history lists have inconsistent lengths"
            )
        saved_epoch_accum = resume_checkpoint.get("epoch_accum", {})
        epoch_accum = {
            "loss_sum": float(saved_epoch_accum.get("loss_sum", 0.0)),
            "sample_count": int(saved_epoch_accum.get("sample_count", 0)),
            "seconds": float(saved_epoch_accum.get("seconds", 0.0)),
        }

    chunks_per_epoch = len(train_idxs)
    completed_epochs_from_chunks = completed_chunks // chunks_per_epoch
    if completed_epochs != completed_epochs_from_chunks:
        raise ValueError(
            f"checkpoint completed_epochs={completed_epochs}, but "
            f"completed_chunks={completed_chunks} implies "
            f"{completed_epochs_from_chunks} completed epoch(s)"
        )
    partial_chunk_pos = completed_chunks % chunks_per_epoch
    if partial_chunk_pos == 0:
        if epoch_accum["sample_count"] != 0:
            raise ValueError(
                "checkpoint has a complete epoch boundary but non-empty epoch_accum"
            )
        epoch_accum = {"loss_sum": 0.0, "sample_count": 0, "seconds": 0.0}
    elif epoch_accum["sample_count"] <= 0:
        raise ValueError(
            "checkpoint stops inside an epoch but does not contain a usable epoch_accum"
        )

    if num_chunks is None:
        if lr is not None:
            epochs_to_complete = 1 if num_epochs is None else num_epochs
        else:
            remaining_epochs = len(paper_lr_schedule) - completed_epochs
            if remaining_epochs < 1:
                raise ValueError(
                    "The paper learning-rate schedule is already complete. "
                    "Provide lr to continue with a custom learning rate."
                )
            epochs_to_complete = (
                remaining_epochs if num_epochs is None else num_epochs
            )
            if epochs_to_complete > remaining_epochs:
                raise ValueError(
                    f"Only {remaining_epochs} paper-schedule epoch(s) remain, "
                    f"but num_epochs={epochs_to_complete} was requested."
                )
        chunks_this_call = epochs_to_complete * chunks_per_epoch - partial_chunk_pos
    else:
        chunks_this_call = num_chunks

    target_chunks = completed_chunks + chunks_this_call
    first_epoch_touched = completed_chunks // chunks_per_epoch + 1
    last_epoch_touched = (target_chunks - 1) // chunks_per_epoch + 1

    def learning_rate_for_epoch(epoch_idx):
        if lr is not None:
            return lr
        schedule_idx = epoch_idx - 1
        if schedule_idx >= len(paper_lr_schedule):
            raise ValueError(
                f"num_chunks reaches epoch {epoch_idx}, but the paper learning-rate "
                f"schedule has only {len(paper_lr_schedule)} epochs. Provide lr to "
                "continue with a custom learning rate."
            )
        return paper_lr_schedule[schedule_idx]

    run_lr_schedule = [
        learning_rate_for_epoch(epoch_idx)
        for epoch_idx in range(first_epoch_touched, last_epoch_touched + 1)
    ]

    print(
        f"device={device}, model_type={model_type}, target_pair={target_pair}, "
        f"target_parameterization={target_parameterization}, dim_eta={dim_eta}, "
        f"hidden_dim={hidden_dim}, n_hidden={n_hidden}"
    )
    print(f"chunk_dir={chunk_dir}, eta_path={eta_path}")
    print(f"train_chunks={len(train_idxs)}, val_chunks={len(val_idxs)}")
    print(
        f"rows/epoch={rows_per_epoch:,}, effective_rows/epoch={effective_rows_per_epoch:,}, "
        f"batch_size={batch_size:,}"
    )
    print(
        f"completed_epochs={completed_epochs}, completed_chunks={completed_chunks}, "
        f"chunks_this_call={chunks_this_call}, target_chunks={target_chunks}"
    )
    print(
        f"val_every_chunks={val_every_chunks}, "
        f"tmp_save_every_chunks={tmp_save_every_chunks}, "
        f"memory_on_gpu={memory_on_gpu}"
    )
    print(f"BN pretraining first {bn_pretrain_fraction:.1%} = {bn_freeze_after_rows:,} rows")
    print(f"run LR schedule={run_lr_schedule}")

    chunk_batch_iterator = (
        iter_device_chunk_batches if memory_on_gpu else iter_chunk_batches
    )

    def chunk_order_for_epoch(epoch_idx):
        order = list(train_idxs)
        if shuffle_chunks:
            order = np.random.default_rng(seed + epoch_idx).permutation(order).tolist()
        return order

    def validate_current_model():
        validation_start = time.perf_counter()
        model.eval()
        vsum, vcount = 0.0, 0
        with torch.no_grad():
            for validation_idx in sorted(val_idxs):
                for x_batch, eta_batch in chunk_batch_iterator(
                    path_by_idx[validation_idx],
                    etas,
                    eta_min,
                    eta_max,
                    batch_size,
                    device,
                    drop_last=False,
                    validate=validate_data,
                ):
                    loss = model.nll(x_batch, eta_batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Validation NLL is NaN/Inf in chunk {validation_idx}"
                        )
                    bsz = x_batch.shape[0]
                    vsum += float(loss) * bsz
                    vcount += bsz
                    del x_batch, eta_batch, loss
        if vcount == 0:
            raise ValueError("Validation chunks produced no samples")
        return vsum / vcount, time.perf_counter() - validation_start

    def checkpoint_payload():
        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "dim_x": 2,
            "dim_eta": dim_eta,
            "n_coupling": model.n_coupling,
            "hidden_dim": model.hidden_dim,
            "n_hidden": model.n_hidden,
            "t_negative_slope": t_slope,
            "use_bn": model.use_bn,
            "scale_clip": scale_clip,
            "eta_min": eta_min,
            "eta_max": eta_max,
            "history": history,
            "chunk_loss_history": chunk_loss_history,
            "validation_history": validation_history,
            "epoch_accum": dict(epoch_accum),
            "lr_schedule": paper_lr_schedule,
            "run_lr_schedule": run_lr_schedule,
            "completed_epochs": completed_epochs,
            "completed_chunks": completed_chunks,
            "num_chunks": num_chunks,
            "val_every_chunks": val_every_chunks,
            "tmp_save_every_chunks": tmp_save_every_chunks,
            "epoch1_rows_seen": epoch1_rows_seen,
            "bn_frozen": bn_has_been_frozen,
            "train_chunk_idxs": train_idxs,
            "validation_chunk_idxs": sorted(val_idxs),
            "bn_pretrain_fraction": bn_pretrain_fraction,
            "paper_preset": "Kim_et_al_2022_architecture_adapted_to_running_minimum",
            "paper_exact_target": False,
            "target_pair": target_pair,
            "target_parameterization": target_parameterization,
            "model_coordinates": list(model.model_coordinates),
            "physical_coordinates": list(model.physical_coordinates),
            "support_parameterization": (
                SUPPORT_PARAMETERIZATION
                if target_parameterization == "ut"
                else "model output M_T"
            ),
            "support_eps": model.support_eps,
            "path_statistic": "running_minimum",
            "data_columns": ["eta_index", "X_T", "running_minimum"],
            "model_type": model_type,
            "chunk_dir": chunk_dir,
            "eta_path": eta_path,
            "batch_size": batch_size,
            "drop_last": bool(drop_last),
            "memory_on_gpu": memory_on_gpu,
            "seed": seed,
        }

    def checkpoint_path_with_suffix(checkpoint_path, suffix):
        checkpoint_path = Path(checkpoint_path)
        return checkpoint_path.with_name(
            f"{checkpoint_path.stem}{suffix}{checkpoint_path.suffix}"
        )

    def save_checkpoint(checkpoint_path=None):
        nonlocal ckpt
        checkpoint_path = save_path if checkpoint_path is None else Path(checkpoint_path)
        ckpt = checkpoint_payload()
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".writing")
        try:
            torch.save(ckpt, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        except Exception:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass
            raise
        print(
            f"checkpoint saved: {checkpoint_path} "
            f"(completed_epochs={completed_epochs}, completed_chunks={completed_chunks})"
        )

    def save_tmp_checkpoint():
        save_checkpoint(tmp_save_path)
        print(
            f"tmp checkpoint saved: {tmp_save_path} | "
            f"completed_chunks={completed_chunks}",
            flush=True,
        )

    def remove_tmp_checkpoint():
        if not tmp_save_path.exists():
            return
        try:
            tmp_save_path.unlink()
            print(f"tmp checkpoint removed: {tmp_save_path}", flush=True)
        except OSError as exc:
            print(f"failed to remove tmp checkpoint: {exc!r}", flush=True)

    ckpt = None
    last_validation_chunk = None
    last_validation_nll = None
    start_completed_chunks = completed_chunks
    train_start = time.perf_counter()
    tmp_save_path = checkpoint_path_with_suffix(save_path, "_tmp")

    try:
        for offset in range(chunks_this_call):
            global_chunk_before = completed_chunks
            epoch_idx = global_chunk_before // chunks_per_epoch + 1
            chunk_pos_zero = global_chunk_before % chunks_per_epoch
            pos = chunk_pos_zero + 1
            order = chunk_order_for_epoch(epoch_idx)
            ci = order[chunk_pos_zero]
            epoch_lr = learning_rate_for_epoch(epoch_idx)
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
    
            chunk_start = time.perf_counter()
            chunk_sample_count = 0
            chunk_loss_sum = 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for x_batch, eta_batch in chunk_batch_iterator(
                path_by_idx[ci],
                etas,
                eta_min,
                eta_max,
                batch_size,
                device,
                drop_last=drop_last,
                validate=validate_data,
            ):
                if (
                    epoch_idx == 1
                    and not bn_has_been_frozen
                    and epoch1_rows_seen >= bn_freeze_after_rows
                ):
                    freeze_batchnorm(model)
                    bn_has_been_frozen = True
                    print(f"[BN frozen] after {epoch1_rows_seen:,} rows")
    
                model.train()
                enforce_frozen_batchnorm(model)
                loss = model.nll(x_batch, eta_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "NLL is NaN/Inf. Inspect the reported HDF5 slice; for a stabilized "
                        "non-paper run, scale_clip=5.0 can be used."
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    
                bsz = x_batch.shape[0]
                loss_value = float(loss.detach())
                chunk_loss_sum += loss_value * bsz
                chunk_sample_count += bsz
                epoch_accum["loss_sum"] += loss_value * bsz
                epoch_accum["sample_count"] += bsz
                if epoch_idx == 1:
                    epoch1_rows_seen += bsz
                del x_batch, eta_batch, loss
    
            if chunk_sample_count == 0:
                raise ValueError(
                    f"chunk {ci}: rows={counts[path_by_idx[ci]]:,} produced no training batch"
                )
    
            chunk_nll = chunk_loss_sum / chunk_sample_count
            epoch_running_nll = (
                epoch_accum["loss_sum"] / epoch_accum["sample_count"]
            )
            chunk_seconds = time.perf_counter() - chunk_start
            epoch_accum["seconds"] += chunk_seconds
            gpu_memory_gib = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda" else 0.0
            )
            completed_chunks += 1
            chunk_loss_history["epoch"].append(epoch_idx)
            chunk_loss_history["global_chunk"].append(completed_chunks)
            chunk_loss_history["chunk_pos"].append(pos)
            chunk_loss_history["chunk_idx"].append(ci)
            chunk_loss_history["chunk_file"].append(os.path.basename(path_by_idx[ci]))
            chunk_loss_history["nll"].append(chunk_nll)
            chunk_loss_history["epoch_running_nll"].append(epoch_running_nll)
            chunk_loss_history["lr"].append(epoch_lr)
            chunk_loss_history["seconds"].append(chunk_seconds)
            chunk_loss_history["gpu_memory_gib"].append(gpu_memory_gib)
            print(
                f"Chunk step {completed_chunks:5d} | epoch {epoch_idx:3d} "
                f"chunk {pos:3d}/{chunks_per_epoch} | idx={ci:3d} | "
                f"NLL={chunk_nll:.6f} | epoch NLL={epoch_running_nll:.6f} | "
                f"time={chunk_seconds:.1f}s | GPU={gpu_memory_gib:.2f} GiB",
                flush=True,
            )
    
            is_epoch_end = completed_chunks % chunks_per_epoch == 0
            is_call_end = offset == chunks_this_call - 1
            should_validate = bool(val_idxs) and (
                (val_every_chunks is not None and completed_chunks % val_every_chunks == 0)
                or is_epoch_end
                or is_call_end
            )
            if should_validate:
                val_nll, validation_seconds = validate_current_model()
                epoch_accum["seconds"] += validation_seconds
                validation_history["global_chunk"].append(completed_chunks)
                validation_history["epoch"].append(epoch_idx)
                validation_history["chunk_pos"].append(pos)
                validation_history["nll"].append(val_nll)
                validation_history["lr"].append(epoch_lr)
                validation_history["seconds"].append(validation_seconds)
                last_validation_chunk = completed_chunks
                last_validation_nll = val_nll
                print(
                    f"Validation @ chunk {completed_chunks:5d} | "
                    f"epoch {epoch_idx:3d} | NLL={val_nll:.6f} | "
                    f"time={validation_seconds:.1f}s",
                    flush=True,
                )
    
            if is_epoch_end:
                if epoch_idx == 1 and not bn_has_been_frozen:
                    freeze_batchnorm(model)
                    bn_has_been_frozen = True
                    print(f"[BN frozen] at end of epoch 1 after {epoch1_rows_seen:,} rows")
    
                train_nll = epoch_accum["loss_sum"] / epoch_accum["sample_count"]
                epoch_val_nll = (
                    last_validation_nll
                    if last_validation_chunk == completed_chunks
                    else None
                )
                epoch_seconds = epoch_accum["seconds"]
                history["epoch_nll"].append(train_nll)
                history["val_nll"].append(epoch_val_nll)
                history["lr"].append(epoch_lr)
                history["epoch_seconds"].append(epoch_seconds)
                completed_epochs = epoch_idx
                print(
                    f"[epoch {epoch_idx}] train={train_nll:.6f}, val={epoch_val_nll}, "
                    f"lr={epoch_lr:.1e}, time={epoch_seconds:.1f}s"
                )
                epoch_accum = {"loss_sum": 0.0, "sample_count": 0, "seconds": 0.0}
                save_checkpoint()
    
            if (
                tmp_save_every_chunks is not None
                and (completed_chunks - start_completed_chunks) % tmp_save_every_chunks == 0
            ):
                save_tmp_checkpoint()
    except (KeyboardInterrupt, Exception) as exc:
        total_seconds = time.perf_counter() - train_start
        print(
            f"\ntraining interrupted ({type(exc).__name__}) | "
            f"completed_chunks={completed_chunks}/{target_chunks}",
            flush=True,
        )
        if completed_chunks > start_completed_chunks:
            interrupt_save_path = checkpoint_path_with_suffix(
                save_path, f"_interrupt{completed_chunks}"
            )
            try:
                save_checkpoint(interrupt_save_path)
                print(
                    f"interrupt checkpoint saved: {interrupt_save_path}",
                    flush=True,
                )
            except Exception as save_exc:
                print(
                    f"failed to save interrupt checkpoint: {save_exc!r}",
                    flush=True,
                )
        else:
            print(
                "No new chunk was completed; no interrupt checkpoint was saved.",
                flush=True,
            )
        print(
            f"time before interruption: {total_seconds / 60:.2f}m "
            f"({total_seconds / 3600:.2f}h)",
            flush=True,
        )
        raise

    if completed_chunks % chunks_per_epoch != 0:
        save_checkpoint()
    remove_tmp_checkpoint()
    total_seconds = time.perf_counter() - train_start
    print(
        f"training complete: {save_path} | "
        f"time={total_seconds / 60:.2f}m ({total_seconds / 3600:.2f}h)"
    )
    model.eval()
    return model, ckpt

def plot_crealnvp_history(ckpt, window=None, figsize=(9, 7)):
    """Plot available epoch, chunk, and chunk-validation NLL histories."""
    import matplotlib.pyplot as plt

    history = ckpt.get("history", {})
    train_nll = np.asarray(history.get("epoch_nll", []), dtype=float)
    val_nll = np.asarray([
        np.nan if value is None else value
        for value in history.get("val_nll", [])
    ], dtype=float)
    if val_nll.size not in (0, train_nll.size):
        raise ValueError("epoch train/validation history lengths are inconsistent")

    chunk_history = ckpt.get("chunk_loss_history", {})
    chunk_nll = np.asarray(chunk_history.get("nll", []), dtype=float)
    validation_history = ckpt.get("validation_history", {})
    chunk_val_nll = np.asarray(validation_history.get("nll", []), dtype=float)
    chunk_val_steps = np.asarray(
        validation_history.get("global_chunk", []), dtype=int
    )
    if chunk_val_steps.size != chunk_val_nll.size:
        raise ValueError("chunk validation index and NLL history lengths are inconsistent")

    has_epoch_history = train_nll.size > 0
    has_chunk_history = chunk_nll.size > 0
    if not has_epoch_history and not has_chunk_history:
        raise ValueError("checkpoint does not contain epoch or chunk NLL history")

    n_rows = int(has_epoch_history) + int(has_chunk_history)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)
    axes = axes[:, 0]
    axis_idx = 0

    if has_epoch_history:
        epoch_axis = axes[axis_idx]
        axis_idx += 1
        epochs = np.arange(1, train_nll.size + 1)
        epoch_axis.plot(epochs, train_nll, marker="o", label="Train NLL")
        if val_nll.size and np.isfinite(val_nll).any():
            epoch_axis.plot(epochs, val_nll, marker="o", label="Validation NLL")
        epoch_axis.set_xlabel("Epoch")
        epoch_axis.set_ylabel("NLL")
        epoch_axis.set_title("CRealNVP Epoch Loss History")
        epoch_axis.set_xticks(epochs)
        epoch_axis.grid(True, alpha=0.3)
        epoch_axis.legend()

    if has_chunk_history:
        chunk_axis = axes[axis_idx]
        global_chunks = np.asarray(
            chunk_history.get(
                "global_chunk",
                np.arange(1, chunk_nll.size + 1),
            ),
            dtype=int,
        )
        if global_chunks.size != chunk_nll.size:
            raise ValueError("chunk global index and NLL history lengths are inconsistent")

        if window is None:
            window = max(1, len(ckpt.get("train_chunk_idxs", [])))
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
            raise TypeError("window must be a positive integer")
        window = int(window)
        if window < 1:
            raise ValueError("window must be >= 1")

        chunk_axis.plot(
            global_chunks,
            chunk_nll,
            color="steelblue",
            linewidth=1,
            alpha=0.35,
            label="Chunk NLL",
        )
        if chunk_nll.size >= window:
            moving_average = np.convolve(
                chunk_nll,
                np.ones(window, dtype=float) / window,
                mode="valid",
            )
            chunk_axis.plot(
                global_chunks[window - 1:],
                moving_average,
                color="navy",
                linewidth=2,
                label=f"Moving average (window={window})",
            )
        if chunk_val_nll.size:
            chunk_axis.plot(
                chunk_val_steps,
                chunk_val_nll,
                color="darkorange",
                marker="o",
                linewidth=1.5,
                label="Validation NLL",
            )
        chunk_axis.set_xlabel("Global Chunk")
        chunk_axis.set_ylabel("NLL")
        chunk_axis.set_title("CRealNVP Chunk Loss History")
        chunk_axis.grid(True, alpha=0.3)
        chunk_axis.legend()

    fig.tight_layout()
    return fig, axes

def load_crealnvp_checkpoint(path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target_parameterization = _checkpoint_target_parameterization(ckpt)
    model = CRealNVP2D(
        dim_eta=ckpt["dim_eta"], n_coupling=ckpt["n_coupling"],
        hidden_dim=ckpt["hidden_dim"], n_hidden=ckpt["n_hidden"],
        t_negative_slope=ckpt["t_negative_slope"], use_bn=ckpt["use_bn"],
        scale_clip=ckpt.get("scale_clip"),
        target_parameterization=target_parameterization,
        support_eps=float(ckpt.get("support_eps", SUPPORT_EPS)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    freeze_batchnorm(model)
    model.eval()
    return model, ckpt


def normalize_eta_tensor(
    eta_raw, ckpt, device, dtype=torch.float32, *, allow_extrapolation=False,
):
    eta = torch.as_tensor(eta_raw, device=device, dtype=dtype)
    if eta.ndim not in {1, 2} or eta.shape[-1] != int(ckpt["dim_eta"]):
        raise ValueError(
            f"eta_raw must have shape ({ckpt['dim_eta']},) or (N, {ckpt['dim_eta']}), "
            f"got {tuple(eta.shape)}"
        )
    eta_min = torch.as_tensor(ckpt["eta_min"], device=device, dtype=dtype)
    eta_max = torch.as_tensor(ckpt["eta_max"], device=device, dtype=dtype)
    span = eta_max - eta_min
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    normalized = (eta - eta_min) / safe_span
    normalized = torch.where(span > 0, normalized, torch.zeros_like(normalized))
    if not allow_extrapolation and ((normalized < -1e-6).any() or (normalized > 1.0 + 1e-6).any()):
        raise ValueError(
            "eta_raw is outside the checkpoint's training range. "
            "Pass allow_extrapolation=True only for an intentional extrapolation experiment."
        )
    return normalized


@torch.no_grad()
def sample_crealnvp(
    model, ckpt, eta_raw, n_samples=100_000, antithetic=False, seed=1234,
    *, allow_extrapolation=False, return_transformed=False,
):
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eta = normalize_eta_tensor(
        eta_raw, ckpt, device, dtype, allow_extrapolation=allow_extrapolation
    )
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return model.sample(
        eta, n_samples, antithetic=antithetic, generator=generator,
        return_transformed=return_transformed,
    )


@torch.no_grad()
def sample_valid_crealnvp(
    model, ckpt, eta_raw, n_samples=100_000, antithetic=False, seed=1234,
    *, allow_extrapolation=False, max_resample_rounds=100, valid_resample=True,
    initial_samples=None,
):
    """Validate physical support; UT is diagnostic-only and never resamples."""
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    max_resample_rounds = int(max_resample_rounds)
    if max_resample_rounds < 1:
        raise ValueError("max_resample_rounds must be >= 1")
    if not isinstance(valid_resample, (bool, np.bool_)):
        raise TypeError("valid_resample must be True or False")
    valid_resample = bool(valid_resample)
    target_parameterization = _checkpoint_target_parameterization(ckpt)

    if target_parameterization == "ut":
        samples = (
            sample_crealnvp(
                model, ckpt, eta_raw, n_samples=n_samples,
                antithetic=antithetic, seed=seed,
                allow_extrapolation=allow_extrapolation,
            )
            if initial_samples is None
            else initial_samples
        )
        if samples.shape != (n_samples, 2):
            raise ValueError(
                f"initial/generated samples must have shape ({n_samples}, 2)."
            )
        invalid_count = int(_support_invalid_mask(samples).sum().item())
        if invalid_count:
            raise RuntimeError(
                "UT NVP produced support-invalid samples: "
                f"{invalid_count:,}/{n_samples:,}. No samples were rejected or resampled."
            )
        diagnostics = {
            "validation_applied": True,
            "target_parameterization": "ut",
            "valid_resample": False,
            "valid_resample_requested": valid_resample,
            "invalid_path_fraction": 0.0,
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
            "support_invalid_count": 0,
        }
        return samples, diagnostics

    valid_batches = []
    accepted_samples = 0
    attempted_samples = 0
    rejected_samples = 0
    sampling_rounds = max_resample_rounds if valid_resample else 1

    for round_idx in range(sampling_rounds):
        remaining = n_samples - accepted_samples
        if remaining <= 0:
            break
        draw_count = remaining
        round_seed = None if seed is None else int(seed) + round_idx
        if round_idx == 0 and initial_samples is not None:
            candidates = initial_samples
            if candidates.shape[0] != n_samples:
                raise ValueError(
                    "initial_samples must contain exactly n_samples rows"
                )
            draw_count = n_samples
        else:
            candidates = sample_crealnvp(
                model,
                ckpt,
                eta_raw,
                n_samples=draw_count,
                antithetic=antithetic,
                seed=round_seed,
                allow_extrapolation=allow_extrapolation,
            )
        invalid = _support_invalid_mask(candidates)
        valid = ~invalid

        attempted_samples += draw_count
        rejected_samples += int(invalid.sum().item())
        accepted = candidates[valid]
        if accepted.numel() > 0:
            accepted = accepted[:remaining]
            valid_batches.append(accepted)
            accepted_samples += int(accepted.shape[0])
        del candidates, invalid, valid, accepted

    if accepted_samples == 0:
        raise RuntimeError(
            f"No valid CRealNVP paths were found in {attempted_samples:,} draws."
        )
    if valid_resample and accepted_samples < n_samples:
        raise RuntimeError(
            f"Could obtain only {accepted_samples:,} valid paths after "
            f"{attempted_samples:,} draws. The model is producing too many "
            "invalid running-minimum samples."
        )

    samples = torch.cat(valid_batches, dim=0)
    regenerated_samples = max(attempted_samples - n_samples, 0)
    discarded_fraction = rejected_samples / attempted_samples
    regenerated_fraction = regenerated_samples / n_samples
    diagnostics = {
        "validation_applied": True,
        "target_parameterization": "mt",
        "valid_resample": valid_resample,
        "valid_resample_requested": valid_resample,
        "invalid_path_fraction": discarded_fraction,
        "rejected_path_count": rejected_samples,
        "total_drawn_samples": attempted_samples,
        "accepted_path_count": accepted_samples,
        "requested_sample_count": n_samples,
        "used_sample_count": accepted_samples,
        "discarded_sample_count": rejected_samples,
        "discarded_sample_fraction": discarded_fraction,
        "discarded_sample_pct": 100.0 * discarded_fraction,
        "regenerated_sample_count": regenerated_samples,
        "regenerated_sample_fraction": regenerated_fraction,
        "regenerated_sample_pct": 100.0 * regenerated_fraction,
        "support_invalid_count": rejected_samples,
    }
    return samples, diagnostics


@torch.no_grad()
def diagnose_crealnvp_samples(
    model, ckpt, eta_raw, n_samples=100_000, antithetic=False, seed=1234,
    *, allow_extrapolation=False, tolerance=SUPPORT_TOLERANCE,
    print_output=True,
):
    model_samples = sample_crealnvp(
        model, ckpt, eta_raw, n_samples=n_samples,
        antithetic=antithetic, seed=seed,
        allow_extrapolation=allow_extrapolation,
        return_transformed=True,
    )
    physical_samples = model.model_to_physical(model_samples)
    X_T = physical_samples[:, 0]
    M_T = physical_samples[:, 1]
    D_T = torch.minimum(torch.zeros_like(X_T), X_T) - M_T
    if _checkpoint_target_parameterization(ckpt) == "ut":
        U_T = model_samples[:, 1]
    else:
        U_T = inverse_softplus_tensor(torch.clamp(D_T, min=SUPPORT_EPS))

    finite_rows = (
        torch.isfinite(X_T) & torch.isfinite(M_T)
        & torch.isfinite(D_T) & torch.isfinite(U_T)
    )
    invalid = _support_invalid_mask(physical_samples, tolerance=tolerance)
    invalid_count = int(invalid.sum().item())
    sample_count = int(physical_samples.shape[0])
    probabilities = [0.001, 0.01, 0.5, 0.99, 0.999]

    def min_max(values):
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return float("nan"), float("nan")
        return float(finite.min().item()), float(finite.max().item())

    def quantiles(values):
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return {str(q): float("nan") for q in probabilities}
        q = torch.as_tensor(
            probabilities, device=finite.device, dtype=finite.dtype,
        )
        values_q = torch.quantile(finite, q)
        return {
            str(probability): float(value.item())
            for probability, value in zip(probabilities, values_q)
        }

    x_min, x_max = min_max(X_T)
    m_min, m_max = min_max(M_T)
    d_min, d_max = min_max(D_T)
    u_min, u_max = min_max(U_T)
    diagnostics = {
        "generated_sample_count": sample_count,
        "target_parameterization": _checkpoint_target_parameterization(ckpt),
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
        "X_T_quantiles": quantiles(X_T),
        "M_T_quantiles": quantiles(M_T),
        "D_T_quantiles": quantiles(D_T),
        "support_tolerance": float(tolerance),
    }
    if print_output:
        print(f"generated sample count : {sample_count:,}")
        print(f"target parameterization: {diagnostics['target_parameterization']}")
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

    if diagnostics["target_parameterization"] == "ut" and invalid_count:
        raise RuntimeError(
            "UT NVP produced a nonzero support-invalid count. "
            "No samples were rejected or resampled."
        )
    return diagnostics


@torch.no_grad()
def price_current_options(
    model, ckpt, eta_raw, *, S0=1.0, K=1.0, B=0.8, r, T,
    n_samples=100_000, antithetic=True, seed=1234, allow_extrapolation=False,
    validate_samples=True, valid_resample=True, mt_corr=False,
):
    """Price options; MT may filter invalid paths, while UT is diagnostic-only."""
    eta_array = np.asarray(eta_raw)
    if eta_array.ndim != 1:
        raise ValueError("price_current_options expects one eta parameter vector")
    if not (math.isclose(float(eta_array[0]), float(r), rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(float(eta_array[-1]), float(T), rel_tol=0.0, abs_tol=1e-7)):
        raise ValueError("r and T must agree with the first and last entries of eta_raw")
    if S0 <= 0 or K <= 0 or B < 0:
        raise ValueError("S0 and K must be positive, and B must be non-negative")
    if B >= S0:
        raise ValueError("A down barrier must satisfy B < S0 because the path starts at S0")
    if not isinstance(validate_samples, (bool, np.bool_)):
        raise TypeError("validate_samples must be True or False")
    if not isinstance(valid_resample, (bool, np.bool_)):
        raise TypeError("valid_resample must be True or False")
    if not isinstance(mt_corr, (bool, np.bool_)):
        raise TypeError("mt_corr must be True or False")

    raw_samples = sample_crealnvp(
        model,
        ckpt,
        eta_raw,
        n_samples=n_samples,
        antithetic=antithetic,
        seed=seed,
        allow_extrapolation=allow_extrapolation,
    )

    if validate_samples:
        barrier_samples, diagnostics = sample_valid_crealnvp(
            model,
            ckpt,
            eta_raw,
            n_samples=n_samples,
            antithetic=antithetic,
            seed=seed,
            allow_extrapolation=allow_extrapolation,
            valid_resample=valid_resample,
            initial_samples=raw_samples,
        )
    else:
        barrier_samples = raw_samples
        diagnostics = {
            "validation_applied": False,
            "valid_resample": False,
            "invalid_path_fraction": None,
            "rejected_path_count": None,
            "total_drawn_samples": int(n_samples),
            "accepted_path_count": None,
            "requested_sample_count": int(n_samples),
            "used_sample_count": int(n_samples),
            "discarded_sample_count": None,
            "discarded_sample_fraction": None,
            "discarded_sample_pct": None,
            "regenerated_sample_count": 0,
            "regenerated_sample_fraction": 0.0,
            "regenerated_sample_pct": 0.0,
        }
    vanilla_X_T = raw_samples[:, 0]
    vanilla_S_T = S0 * torch.exp(vanilla_X_T)
    barrier_X_T = barrier_samples[:, 0]
    running_min = barrier_samples[:, 1]
    if mt_corr:
        mt_upper = torch.minimum(torch.zeros_like(barrier_X_T), barrier_X_T)
        corrected_mask = running_min > mt_upper
        running_min = torch.minimum(running_min, mt_upper)
    else:
        corrected_mask = torch.zeros_like(running_min, dtype=torch.bool)
    corrected_count = int(corrected_mask.sum().item())
    corrected_fraction = corrected_count / int(running_min.shape[0])
    diagnostics.update({
        "mt_corr_applied": bool(mt_corr),
        "mt_corrected_sample_count": corrected_count,
        "mt_corrected_sample_fraction": corrected_fraction,
        "mt_corrected_sample_pct": 100.0 * corrected_fraction,
    })
    barrier_S_T = S0 * torch.exp(barrier_X_T)
    min_S = S0 * torch.exp(running_min)
    if validate_samples and not (
        torch.isfinite(barrier_S_T).all() and torch.isfinite(min_S).all()
    ):
        raise FloatingPointError(
            "A physically valid generated path overflowed during exp(X). "
            "The model's right tail is not stable enough for option pricing."
        )

    vanilla_call = torch.clamp(vanilla_S_T - K, min=0.0)
    vanilla_put = torch.clamp(K - vanilla_S_T, min=0.0)
    barrier_call = torch.clamp(barrier_S_T - K, min=0.0)
    barrier_put = torch.clamp(K - barrier_S_T, min=0.0)
    alive = (min_S > B).to(barrier_samples.dtype)
    discount = math.exp(-float(r) * float(T))
    down_out_call = discount * (barrier_call * alive).mean().item()
    down_out_put = discount * (barrier_put * alive).mean().item()
    return {
        "van_call": discount * vanilla_call.mean().item(),
        "van_put": discount * vanilla_put.mean().item(),
        "down_out_call": down_out_call,
        "down_out_put": down_out_put,
        # Backward-compatible aliases; these are specifically down-and-out prices.
        "barr_call": down_out_call,
        "barr_put": down_out_put,
        "vanilla_sample_count": int(raw_samples.shape[0]),
        **diagnostics,
    }


if __name__ == "__main__":
    SAVE_PATH = "crealnvp_heston_XT_MIN_paper2022_adapted.pt"

    model, ckpt = train_crealnvp_paper2022(
        save_path=SAVE_PATH,
        model_type="hes",             # resolves the current /mnt/d paths automatically
        target_pair="XT_MIN",         # paths[:, 2] is X.min(axis=1), not the paper's maximum
        target_parameterization="mt", # use "ut" for support-aware [X_T, U_T] training
        batch_size=16384,              # the paper does not report its exact batch size
        validation_chunk_idxs=[15,24,78],
        scale_clip=None,               # set 5.0 only for an explicitly stabilized run
        bn_pretrain_fraction=0.05,     # Kim et al. (2022)
        drop_last=True,
        validate_data=True,
    )

    eta_heston = [0.03, 2.0, 0.05, 0.5, -0.7, 0.05, 1.5]
    print(price_current_options(
        model, ckpt, eta_heston,
        S0=1.0, K=1.0, B=0.8, r=0.03, T=1.5,
        n_samples=100_000, antithetic=True, seed=1234,
    ))