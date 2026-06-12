import os
import re
import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from e_1_run_cvae import (
    BS_CHUNK_DIR, BS_ETA_PATH,
    HES_CHUNK_DIR, HES_ETA_PATH,
    compute_eta_stats,
)

from e_2_CVAE import CVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chunk_file_idx(chunk_path):
    filename = os.path.basename(chunk_path)
    match = re.search(r"chunk_(\d+)", filename)
    if match is None:
        raise ValueError(f"Cannot parse chunk index from filename: {filename}")
    return int(match.group(1))


def get_paths_and_eta_path(model_type):
    if model_type == "bs":
        return BS_CHUNK_DIR, BS_ETA_PATH
    elif model_type == "hes":
        return HES_CHUNK_DIR, HES_ETA_PATH
    else:
        raise ValueError("model_type must be 'bs' or 'hes'")


def get_chunk_path_by_idx(model_type):
    chunk_dir, _ = get_paths_and_eta_path(model_type)
    chunk_paths = sorted(
        glob.glob(os.path.join(chunk_dir, "*.h5")),
        key=_chunk_file_idx
    )
    return {_chunk_file_idx(p): p for p in chunk_paths}



def _split_mu_lv(output, name="network output"):
    """
    network output이 tuple(mu, lv)이면 그대로 쓰고,
    tensor이면 마지막 dimension을 반으로 나눠서 mu/lv로 사용.
    """
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output[0], output[1]

    if torch.is_tensor(output):
        if output.shape[-1] % 2 != 0:
            raise ValueError(f"{name} last dim must be even, got {output.shape}")
        return output.chunk(2, dim=-1)

    raise TypeError(f"Cannot parse {name}: {type(output)}")


def get_recognition_prior_params(cvae, x, eta):
    """
    네 CVAE class의 이름이 encoder/prior 또는 recognition/prior일 가능성을 모두 처리.
    반환:
        mu_q, lv_q, mu_p, lv_p
    """

    # recognition / encoder
    q_candidates = [
        "encoder",
        "recognition",
        "recognition_net",
        "q_net",
        "q",
        "encode",
    ]

    q_out = None
    q_used = None

    for name in q_candidates:
        if not hasattr(cvae, name):
            continue

        obj = getattr(cvae, name)

        # 1) encoder(torch.cat([x, eta], dim=1)) 형태
        try:
            q_out = obj(torch.cat([x, eta], dim=1))
            q_used = name
            break
        except TypeError:
            pass

        # 2) encoder(x, eta) 형태
        try:
            q_out = obj(x, eta)
            q_used = name
            break
        except TypeError:
            pass

    if q_out is None:
        raise AttributeError(
            "CVAE에서 encoder/recognition network를 찾지 못했습니다. "
            "CVAE class의 recognition network 이름을 확인해서 get_recognition_prior_params()에 추가하세요."
        )

    mu_q, lv_q = _split_mu_lv(q_out, name=q_used)

    # prior
    p_candidates = [
        "prior",
        "prior_net",
        "p_net",
        "p",
    ]

    p_out = None
    p_used = None

    for name in p_candidates:
        if not hasattr(cvae, name):
            continue

        obj = getattr(cvae, name)

        try:
            p_out = obj(eta)
            p_used = name
            break
        except TypeError:
            pass

    if p_out is None:
        raise AttributeError(
            "CVAE에서 prior network를 찾지 못했습니다. "
            "CVAE class의 prior network 이름을 확인해서 get_recognition_prior_params()에 추가하세요."
        )

    mu_p, lv_p = _split_mu_lv(p_out, name=p_used)

    return mu_q, lv_q, mu_p, lv_p


def load_cvae_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    dim_x = int(ckpt.get("dim_x", 2))
    dim_eta = int(ckpt["dim_eta"])
    dim_z = int(ckpt["dim_z"])
    hidden_dims = ckpt["hidden_dims"]
    use_bn = bool(ckpt.get("use_bn", False))
    x_mean = ckpt.get("x_mean", np.zeros(dim_x, dtype=np.float32))
    x_std = ckpt.get("x_std", np.ones(dim_x, dtype=np.float32))

    cvae = CVAE(
        dim_x=dim_x,
        dim_eta=dim_eta,
        dim_z=dim_z,
        hidden_dims=hidden_dims,
        use_bn=use_bn,
        x_mean=x_mean,
        x_std=x_std,
    ).to(device)

    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    cvae.load_state_dict(state_dict)
    cvae.eval()

    return cvae, ckpt


@torch.no_grad()
def collect_z_white_hist_from_chunks(
    checkpoint_path,
    model_type="bs",
    chunk_idxs=(98, 99),
    mode="sample",
    bins=120,
    xlim=(-5, 5),
    ylim=(-5, 5),
    row_batch_size=262_144,
):
    """
    validation chunk 전체를 하나씩 읽어서 z_white 2D histogram과 진단 통계만 누적한다.
    전체 X/M 또는 z_white 배열을 메모리에 저장하지 않는다.

    mode:
        "sample" : z_q = mu_q + sigma_q * eps 를 prior 기준으로 표준화
        "mean"   : mu_q 자체를 prior 기준으로 표준화
    """
    cvae, ckpt = load_cvae_from_checkpoint(checkpoint_path)

    if int(ckpt["dim_z"]) != 2:
        raise ValueError(f"이 plotting 코드는 dim_z=2용입니다. checkpoint dim_z={ckpt['dim_z']}")
    if mode not in ("sample", "mean"):
        raise ValueError("mode must be 'sample' or 'mean'")

    chunk_path_by_idx = get_chunk_path_by_idx(model_type)
    _, eta_path = get_paths_and_eta_path(model_type)
    etas, eta_min, eta_max = compute_eta_stats(eta_path)

    chunk_idxs = [int(ci) for ci in chunk_idxs]
    missing = [ci for ci in chunk_idxs if ci not in chunk_path_by_idx]
    if missing:
        raise ValueError(f"chunk index not found: {missing}")

    x_edges = np.linspace(xlim[0], xlim[1], int(bins) + 1)
    y_edges = np.linspace(ylim[0], ylim[1], int(bins) + 1)
    counts = np.zeros((int(bins), int(bins)), dtype=np.float64)

    dim_z = int(ckpt["dim_z"])
    n_total = 0
    z_sum = np.zeros(dim_z, dtype=np.float64)
    z_sumsq = np.zeros(dim_z, dtype=np.float64)
    mu_gap_abs_sum = np.zeros(dim_z, dtype=np.float64)
    std_ratio_sum = np.zeros(dim_z, dtype=np.float64)

    x_mean = cvae.x_mean.detach().cpu().numpy().astype(np.float32)
    x_std = cvae.x_std.detach().cpu().numpy().astype(np.float32)

    for ci in chunk_idxs:
        chunk_path = chunk_path_by_idx[ci]
        with h5py.File(chunk_path, "r") as f:
            dset = f["paths"]
            if dset.shape[1] < 3:
                raise ValueError(f"{os.path.basename(chunk_path)} must contain [ori_idx, X_T, M_T]")

            for start in range(0, dset.shape[0], row_batch_size):
                rows = dset[start:start + row_batch_size]
                ori_idx = rows[:, 0].astype(int)
                x_np = rows[:, 1:3].astype(np.float32)
                x_np = (x_np - x_mean) / (x_std + 1e-8)

                eta_np = etas[ori_idx].astype(np.float32)
                eta_np = (eta_np - eta_min) / (eta_max - eta_min + 1e-8)

                x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
                eta = torch.as_tensor(eta_np, dtype=torch.float32, device=device)

                mu_q, lv_q, mu_p, lv_p = get_recognition_prior_params(cvae, x, eta)
                std_q = torch.exp(0.5 * lv_q)
                std_p = torch.exp(0.5 * lv_p).clamp_min(1e-8)

                if mode == "sample":
                    eps = torch.randn_like(mu_q)
                    z_q = mu_q + std_q * eps
                else:
                    z_q = mu_q

                z_white = (z_q - mu_p) / std_p
                mu_gap = (mu_q - mu_p) / std_p
                std_ratio = std_q / std_p

                z_np = z_white.detach().cpu().numpy()
                finite = np.isfinite(z_np[:, 0]) & np.isfinite(z_np[:, 1])
                z_np = z_np[finite]
                if len(z_np) == 0:
                    continue

                hist, _, _ = np.histogram2d(
                    z_np[:, 0],
                    z_np[:, 1],
                    bins=[x_edges, y_edges],
                )
                counts += hist

                mu_gap_np = mu_gap.detach().cpu().numpy()[finite]
                std_ratio_np = std_ratio.detach().cpu().numpy()[finite]
                n = len(z_np)
                n_total += n
                z_sum += z_np.sum(axis=0)
                z_sumsq += (z_np ** 2).sum(axis=0)
                mu_gap_abs_sum += np.abs(mu_gap_np).sum(axis=0)
                std_ratio_sum += std_ratio_np.sum(axis=0)

                del x, eta, mu_q, lv_q, mu_p, lv_p, std_q, std_p, z_q, z_white, mu_gap, std_ratio
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        print(f"processed chunk_idx={ci} | cumulative n={n_total}")

    if n_total == 0:
        raise ValueError("no finite z_white values collected")

    hist_total = float(counts.sum())
    if hist_total == 0:
        raise ValueError("no z_white values inside the requested xlim/ylim range")

    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    density = counts / (hist_total * dx * dy)
    z_mean = z_sum / n_total
    z_var = np.maximum(z_sumsq / n_total - z_mean ** 2, 0.0)

    stats = {
        "checkpoint_path": checkpoint_path,
        "dim_z": dim_z,
        "use_bn": bool(ckpt.get("use_bn", False)),
        "trained_chunks": int(ckpt.get("trained_chunks", -1)),
        "chunk_idxs": chunk_idxs,
        "n_samples": int(n_total),
        "hist_samples": int(hist_total),
        "z_white_mean": z_mean,
        "z_white_std": np.sqrt(z_var),
        "mu_gap_mean_abs": mu_gap_abs_sum / n_total,
        "std_ratio_mean": std_ratio_sum / n_total,
    }

    return density, x_edges, y_edges, stats

def plot_z_white_fig4_style(
    checkpoint_paths,
    labels,
    model_type="bs",
    chunk_idxs=(15, 24, 78),
    mode="sample",
    bins=120,
    xlim=(-5, 5),
    ylim=(-5, 5),
    row_batch_size=2**16,
    title=None,
    save_path=None,
):

    if len(checkpoint_paths) != len(labels):
        raise ValueError("checkpoint_paths와 labels 길이가 같아야 합니다.")

    n_cols = len(checkpoint_paths)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    all_stats = []

    for ax, ckpt_path, label in zip(axes, checkpoint_paths, labels):
        density, x_edges, y_edges, stats = collect_z_white_hist_from_chunks(
            checkpoint_path=ckpt_path,
            model_type=model_type,
            chunk_idxs=chunk_idxs,
            mode=mode,
            bins=bins,
            xlim=xlim,
            ylim=ylim,
            row_batch_size=row_batch_size,
        )

        all_stats.append(stats)

        ax.imshow(
            density.T,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="viridis",
        )

        ax.set_title(label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\tilde z_1$")
        ax.set_ylabel(r"$\tilde z_2$")

        # 표준정규 기준 원형 범위 참고선
        circle1 = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.0)
        circle2 = plt.Circle((0, 0), 2.0, fill=False, linewidth=1.0, linestyle="--")
        ax.add_patch(circle1)
        ax.add_patch(circle2)

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    for s in all_stats:
        print("\ncheckpoint:", s["checkpoint_path"])
        print("use_bn:", s["use_bn"], "| trained_chunks:", s["trained_chunks"], "| n:", s["n_samples"], "| plotted:", s["hist_samples"])
        print("chunk_idxs:", s["chunk_idxs"])
        print("z_white mean:", np.round(s["z_white_mean"], 4))
        print("z_white std :", np.round(s["z_white_std"], 4))
        print("mean abs mu_gap:", np.round(s["mu_gap_mean_abs"], 4))
        print("mean std_ratio :", np.round(s["std_ratio_mean"], 4))

    return all_stats