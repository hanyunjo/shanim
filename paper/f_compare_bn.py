import os
import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from e_2_CVAE import CVAE


# =================
# shared utilities
# =================
def _selected_chunk_paths(chunk_dir, num_chunks, pattern='*.h5'):
    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, pattern)))

    if len(chunk_paths) == 0:
        raise FileNotFoundError(f'No chunk files found in {chunk_dir} with pattern={pattern}')
    if num_chunks < 1:
        raise ValueError('num_chunks must be >= 1')
    if num_chunks > len(chunk_paths):
        raise ValueError(f'num_chunks={num_chunks} is larger than available chunks={len(chunk_paths)}')

    selected_paths = chunk_paths[:num_chunks]
    print(f'Using real data from {num_chunks}/{len(chunk_paths)} chunks')
    print(f'first chunk: {os.path.basename(selected_paths[0])}')
    print(f'last chunk : {os.path.basename(selected_paths[-1])}')
    return selected_paths


def _merge_stats(n, mean, m2, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    batch_n = len(values)
    if batch_n == 0:
        return n, mean, m2

    batch_mean = float(values.mean())
    batch_m2 = float(((values - batch_mean) ** 2).sum())

    if n == 0:
        return batch_n, batch_mean, batch_m2

    total_n = n + batch_n
    delta = batch_mean - mean
    mean = mean + delta * batch_n / total_n
    m2 = m2 + batch_m2 + delta * delta * n * batch_n / total_n
    return total_n, mean, m2


def _stats_from_array(x, chunk_size=1_000_000):
    x = np.asarray(x)
    n = 0
    mean = 0.0
    m2 = 0.0

    for start in range(0, len(x), chunk_size):
        n, mean, m2 = _merge_stats(n, mean, m2, x[start:start + chunk_size])

    if n == 0:
        raise ValueError('no finite values for statistics')

    var = m2 / n
    return {'n': n, 'mean': mean, 'std': float(np.sqrt(var)), 'var': var}


def _finite_range_1d(arrays):
    lo = np.inf
    hi = -np.inf

    for x in arrays:
        x = np.asarray(x)
        finite = np.isfinite(x)
        if not finite.any():
            continue
        lo = min(lo, float(np.min(x[finite])))
        hi = max(hi, float(np.max(x[finite])))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError('no finite values to plot')
    if lo == hi:
        pad = max(abs(lo) * 1e-6, 1e-6)
        lo -= pad
        hi += pad

    return lo, hi


def _hist1d_all(x, bins=80, value_range=None, chunk_size=1_000_000):
    x = np.asarray(x)
    if value_range is None:
        value_range = _finite_range_1d([x])

    counts = np.zeros(bins, dtype=np.float64)
    total = 0

    for start in range(0, len(x), chunk_size):
        batch = x[start:start + chunk_size]
        batch = batch[np.isfinite(batch)]
        if len(batch) == 0:
            continue
        hist, edges = np.histogram(batch, bins=bins, range=value_range)
        counts += hist
        total += len(batch)

    if total == 0:
        raise ValueError('no finite values to plot')

    widths = np.diff(edges)
    density = counts / (total * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density, total


def _hist2d_all(x, y, bins=80, value_range=None, chunk_size=500_000):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError('x and y must have the same length')

    bins_xy = (bins, bins) if isinstance(bins, int) else bins
    if value_range is None:
        value_range = [_finite_range_1d([x]), _finite_range_1d([y])]

    counts = np.zeros((bins_xy[0], bins_xy[1]), dtype=np.float64)
    total = 0

    for start in range(0, len(x), chunk_size):
        xb = x[start:start + chunk_size]
        yb = y[start:start + chunk_size]
        finite = np.isfinite(xb) & np.isfinite(yb)
        xb = xb[finite]
        yb = yb[finite]
        if len(xb) == 0:
            continue
        hist, x_edges, y_edges = np.histogram2d(xb, yb, bins=bins_xy, range=value_range)
        counts += hist
        total += len(xb)

    if total == 0:
        raise ValueError('no finite values to plot')

    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    density = counts / (total * dx * dy)
    return density, x_edges, y_edges, total


def _as_nonempty_array(x):
    if x is None:
        return None
    x = np.asarray(x)
    return x if len(x) > 0 else None


# =================
# statistics / plots
# =================
def statics_result(name, x):
    stats = _stats_from_array(x)

    print()
    print(name)
    print(f'n    : {stats["n"]}')
    print(f'mean : {stats["mean"]:.6f}')
    print(f'std  : {stats["std"]:.6f}')
    print(f'var  : {stats["var"]:.6f}')
    return stats


def plot_1d(name, x, bins=80):
    centers, density, total = _hist1d_all(x, bins=bins)

    plt.figure(figsize=(7, 4))
    plt.plot(centers, density, linewidth=1.8, label=name)
    plt.fill_between(centers, density, alpha=0.25)
    plt.xlabel(name)
    plt.ylabel('Density')
    plt.title(f'{name} distribution (n={total})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_2d_density(x, y, title='2D density', xlabel='X_T', ylabel='M_T', bins=80, cmap='viridis'):
    y = _as_nonempty_array(y)
    if y is None:
        raise ValueError('y must be given for 2D density plot')

    density, x_edges, y_edges, total = _hist2d_all(x, y, bins=bins)

    plt.figure(figsize=(5.5, 5))
    img = plt.imshow(
        density.T,
        origin='lower',
        aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} (n={total})')
    plt.colorbar(img, label='Density')
    plt.tight_layout()
    plt.show()


def plot_three_distributions(real, no_bn, bn, name='X_T', bins=80):
    value_range = _finite_range_1d([real, no_bn, bn])
    real_centers, real_density, real_total = _hist1d_all(real, bins=bins, value_range=value_range)
    no_bn_centers, no_bn_density, no_bn_total = _hist1d_all(no_bn, bins=bins, value_range=value_range)
    bn_centers, bn_density, bn_total = _hist1d_all(bn, bins=bins, value_range=value_range)

    plt.figure(figsize=(8, 5))
    plt.plot(real_centers, real_density, linewidth=1.8, label=f'Real {name} (n={real_total})')
    plt.plot(no_bn_centers, no_bn_density, linewidth=1.8, label=f'CVAE no BN {name} (n={no_bn_total})')
    plt.plot(bn_centers, bn_density, linewidth=1.8, label=f'CVAE BN {name} (n={bn_total})')
    plt.xlabel(name)
    plt.ylabel('Density')
    plt.title(f'{name} distribution comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()





# =================
# real data
# =================
def load_real_xt_mt_from_chunks(chunk_dir, num_chunks, pattern='*.h5', row_batch_size=1_000_000):
    chunk_paths = _selected_chunk_paths(chunk_dir, num_chunks, pattern=pattern)
    x_list = []
    m_list = []
    has_m = None

    for chunk_path in chunk_paths:
        with h5py.File(chunk_path, 'r') as f:
            dset = f['paths']
            if dset.shape[1] < 2:
                raise ValueError(f'{os.path.basename(chunk_path)} does not contain X_T')
            if has_m is None:
                has_m = dset.shape[1] >= 3

            for start in range(0, dset.shape[0], row_batch_size):
                values = dset[start:start + row_batch_size]
                x_list.append(values[:, 1].astype(np.float32))
                if has_m:
                    m_list.append(values[:, 2].astype(np.float32))

    if len(x_list) == 0:
        raise ValueError('no real data loaded')

    real_x = np.concatenate(x_list)
    real_m = np.concatenate(m_list) if has_m and len(m_list) > 0 else None
    print(f'Loaded real data from {len(chunk_paths)} chunks')
    print(f'n_samples : {len(real_x)}')
    return real_x, real_m


def _finalize_stats(n, mean, m2):
    if n == 0:
        raise ValueError('no finite values for statistics')
    var = m2 / n
    return {'n': n, 'mean': mean, 'std': float(np.sqrt(var)), 'var': var}


def _padded_range(lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError('no finite values to plot')
    if lo == hi:
        pad = max(abs(lo) * 1e-6, 1e-6)
        lo -= pad
        hi += pad
    return lo, hi


def _iter_real_chunk_batches(chunk_paths, row_batch_size=1_000_000):
    has_m = None
    for chunk_path in chunk_paths:
        with h5py.File(chunk_path, 'r') as f:
            dset = f['paths']
            if dset.shape[1] < 2:
                raise ValueError(f'{os.path.basename(chunk_path)} does not contain X_T')
            if has_m is None:
                has_m = dset.shape[1] >= 3

            for start in range(0, dset.shape[0], row_batch_size):
                values = dset[start:start + row_batch_size]
                x = values[:, 1].astype(np.float32)
                m = values[:, 2].astype(np.float32) if has_m else None
                yield x, m


def _real_stats_and_ranges_from_chunks(chunk_paths, row_batch_size=1_000_000):
    x_n, x_mean, x_m2 = 0, 0.0, 0.0
    m_n, m_mean, m_m2 = 0, 0.0, 0.0
    x_lo, x_hi = np.inf, -np.inf
    m_lo, m_hi = np.inf, -np.inf

    for x, m in _iter_real_chunk_batches(chunk_paths, row_batch_size=row_batch_size):
        x_n, x_mean, x_m2 = _merge_stats(x_n, x_mean, x_m2, x)
        finite_x = np.isfinite(x)
        if finite_x.any():
            xf = x[finite_x]
            x_lo = min(x_lo, float(xf.min()))
            x_hi = max(x_hi, float(xf.max()))

        if m is not None:
            m_n, m_mean, m_m2 = _merge_stats(m_n, m_mean, m_m2, m)
            finite_m = np.isfinite(m)
            if finite_m.any():
                mf = m[finite_m]
                m_lo = min(m_lo, float(mf.min()))
                m_hi = max(m_hi, float(mf.max()))

    x_stats = _finalize_stats(x_n, x_mean, x_m2)
    m_stats = _finalize_stats(m_n, m_mean, m_m2) if m_n > 0 else None
    x_range = _padded_range(x_lo, x_hi)
    m_range = _padded_range(m_lo, m_hi) if m_n > 0 else None
    return x_stats, m_stats, x_range, m_range


def _real_hists_from_chunks(chunk_paths, bins=80, x_range=None, m_range=None, row_batch_size=1_000_000):
    bins_xy = (bins, bins) if isinstance(bins, int) else bins
    x_bins = bins if isinstance(bins, int) else bins[0]
    m_bins = bins if isinstance(bins, int) else bins[1]
    x_counts = np.zeros(x_bins, dtype=np.float64)
    m_counts = np.zeros(m_bins, dtype=np.float64) if m_range is not None else None
    xy_counts = np.zeros((bins_xy[0], bins_xy[1]), dtype=np.float64) if m_range is not None else None
    x_total = 0
    m_total = 0
    xy_total = 0
    x_edges = None
    m_edges = None
    xy_x_edges = None
    xy_y_edges = None

    for x, m in _iter_real_chunk_batches(chunk_paths, row_batch_size=row_batch_size):
        finite_x = np.isfinite(x)
        xf = x[finite_x]
        if len(xf) > 0:
            hist, x_edges = np.histogram(xf, bins=x_bins, range=x_range)
            x_counts += hist
            x_total += len(xf)

        if m is not None and m_range is not None:
            finite_m = np.isfinite(m)
            mf = m[finite_m]
            if len(mf) > 0:
                hist, m_edges = np.histogram(mf, bins=m_bins, range=m_range)
                m_counts += hist
                m_total += len(mf)

            paired = finite_x & finite_m
            if paired.any():
                hist, xy_x_edges, xy_y_edges = np.histogram2d(
                    x[paired],
                    m[paired],
                    bins=bins_xy,
                    range=[x_range, m_range],
                )
                xy_counts += hist
                xy_total += int(paired.sum())

    return x_counts, x_edges, x_total, m_counts, m_edges, m_total, xy_counts, xy_x_edges, xy_y_edges, xy_total

def _density_1d_from_counts(counts, edges, total):
    if total == 0:
        raise ValueError('no finite values to plot')
    widths = np.diff(edges)
    density = counts / (total * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def _density_2d_from_counts(counts, x_edges, y_edges, total):
    if total == 0:
        raise ValueError('no finite paired values to plot')
    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    return counts / (total * dx * dy)


def _print_stats(name, stats):
    print()
    print(name)
    print(f'n    : {stats["n"]}')
    print(f'mean : {stats["mean"]:.6f}')
    print(f'std  : {stats["std"]:.6f}')
    print(f'var  : {stats["var"]:.6f}')


def show_real_chunk_results(
    chunk_dir,
    num_chunks,
    bins=80,
    pattern='*.h5',
    row_batch_size=1_000_000,
    x_range=None,
    m_range=None,
):
    chunk_paths = _selected_chunk_paths(chunk_dir, num_chunks, pattern=pattern)

    # statics
    real_x_stats, real_m_stats, inferred_x_range, inferred_m_range = _real_stats_and_ranges_from_chunks(
        chunk_paths,
        row_batch_size=row_batch_size,
    )
    _print_stats('Real X_T', real_x_stats)
    if real_m_stats is not None:
        _print_stats('Real M_T', real_m_stats)

    x_range = inferred_x_range if x_range is None else x_range
    m_range = inferred_m_range if m_range is None else m_range

    hist_data = _real_hists_from_chunks(
        chunk_paths,
        bins=bins,
        x_range=x_range,
        m_range=m_range,
        row_batch_size=row_batch_size,
    )
    x_counts, x_edges, x_total, m_counts, m_edges, m_total, xy_counts, xy_x_edges, xy_y_edges, xy_total = hist_data



    x_centers, x_density = _density_1d_from_counts(x_counts, x_edges, x_total)
    plt.figure(figsize=(7, 4))
    plt.plot(x_centers, x_density, linewidth=1.8, label='Real X_T')
    #plt.bar(x_centers, x_density, width=np.diff(x_edges), alpha=0.7) # histogram
    plt.fill_between(x_centers, x_density, alpha=0.25)
    plt.xlabel('X_T')
    plt.ylabel('Density')
    plt.title(f'Real X_T distribution (n={x_total})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if real_m_stats is not None:
        m_centers, m_density = _density_1d_from_counts(m_counts, m_edges, m_total)
        plt.figure(figsize=(7, 4))
        plt.plot(m_centers, m_density, linewidth=1.8, label='Real M_T')
        plt.fill_between(m_centers, m_density, alpha=0.25)
        plt.xlabel('M_T')
        plt.ylabel('Density')
        plt.title(f'Real M_T distribution (n={m_total})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        xy_density = _density_2d_from_counts(xy_counts, xy_x_edges, xy_y_edges, xy_total)
        plt.figure(figsize=(5.5, 5))
        img = plt.imshow(
            xy_density.T,
            origin='lower',
            aspect='auto',
            extent=[xy_x_edges[0], xy_x_edges[-1], xy_y_edges[0], xy_y_edges[-1]],
            cmap='viridis',
        )
        plt.xlabel('X_T')
        plt.ylabel('M_T')
        plt.title(f'Real 2D density (n={xy_total})')
        plt.colorbar(img, label='Density')
        plt.tight_layout()
        plt.show()

    return {
        'real_x': None,
        'real_m': None,
        'real_x_stats': real_x_stats,
        'real_m_stats': real_m_stats,
        'x_hist': {'centers': x_centers, 'density': x_density, 'edges': x_edges, 'counts': x_counts, 'n': x_total},
        'm_hist': None if real_m_stats is None else {'centers': m_centers, 'density': m_density, 'edges': m_edges, 'counts': m_counts, 'n': m_total},
        'xy_hist': None if real_m_stats is None else {'density': xy_density, 'x_edges': xy_x_edges, 'y_edges': xy_y_edges, 'counts': xy_counts, 'n': xy_total},
        'x_range': x_range,
        'm_range': m_range,
        'n_samples': real_x_stats['n'],
    }






# =================
# CVAE samples
# =================
def load_cvae_model(save_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    try:
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(save_path, map_location=device)

    cvae = CVAE(
        dim_x=ckpt['dim_x'],
        dim_eta=ckpt['dim_eta'],
        dim_z=ckpt['dim_z'],
        hidden_dims=ckpt['hidden_dims'],
        use_bn=ckpt.get('use_bn', False),
    ).to(device)

    state_dict = ckpt['model_state']
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    cvae.load_state_dict(state_dict)
    cvae.eval()
    return cvae, ckpt, device


def _load_eta_table(eta_path):
    with h5py.File(os.path.expanduser(eta_path), 'r') as f:
        return f['etas'][:].astype(np.float32)


@torch.no_grad()
def _sample_cvae_for_eta_batch(cvae, eta_batch, ckpt, device):
    eta_scaled = (eta_batch - ckpt['eta_min']) / (ckpt['eta_max'] - ckpt['eta_min'] + 1e-8)
    eta_t = torch.as_tensor(eta_scaled, dtype=torch.float32, device=device)

    mu_p, lv_p = cvae.prior(eta_t)
    z = cvae.reparameterize(mu_p, lv_p, torch.randn_like(mu_p))
    mu_x, lv_x = cvae.decoder(z, eta_t)
    samples = cvae.reparameterize(mu_x, lv_x, torch.randn_like(mu_x))
    return samples.detach().cpu().numpy()


def _iter_cvae_samples_from_chunk_etas(
    cvae,
    ckpt,
    device,
    eta_table,
    chunk_paths,
    row_batch_size=50_000,
):
    for chunk_path in chunk_paths:
        with h5py.File(chunk_path, 'r') as f:
            dset = f['paths']
            for start in range(0, dset.shape[0], row_batch_size):
                stop = min(start + row_batch_size, dset.shape[0])
                ori_idx = dset[start:stop, 0].astype(np.int64)
                eta_batch = eta_table[ori_idx]
                yield _sample_cvae_for_eta_batch(cvae, eta_batch, ckpt, device)


@torch.no_grad()
def load_cvae_xt_mt_from_chunks(
    save_path,
    eta_path,
    chunk_dir,
    num_chunks,
    pattern='*.h5',
    row_batch_size=50_000,
    barr_type='barr',
    device=None,
):
    chunk_paths = _selected_chunk_paths(chunk_dir, num_chunks, pattern=pattern)
    eta_table = _load_eta_table(eta_path)
    cvae, ckpt, device = load_cvae_model(save_path, device=device)

    x_list = []
    m_list = []
    need_m = barr_type == 'barr'
    if need_m and ckpt['dim_x'] < 2:
        raise ValueError('barrier output requires CVAE dim_x >= 2')

    for samples in _iter_cvae_samples_from_chunk_etas(
        cvae,
        ckpt,
        device,
        eta_table,
        chunk_paths,
        row_batch_size=row_batch_size,
    ):
        x_list.append(samples[:, 0].astype(np.float32))
        if need_m:
            m_list.append(samples[:, 1].astype(np.float32))

    if len(x_list) == 0:
        raise ValueError('no CVAE samples generated')

    x_gen = np.concatenate(x_list)
    m_gen = np.concatenate(m_list) if need_m and len(m_list) > 0 else None

    print(f'Generated CVAE samples from {os.path.basename(save_path)}')
    print(f'use_bn    : {ckpt.get("use_bn", False)}')
    print(f'n_samples : {len(x_gen)}')
    return x_gen, m_gen, ckpt


@torch.no_grad()
def load_cvae_xt_mt(save_path, test_etas, n_samples=100000, device=None):
    cvae, ckpt, device = load_cvae_model(save_path, device=device)

    eta_raw = np.array(test_etas, dtype=np.float32)
    eta_scaled = (eta_raw - ckpt['eta_min']) / (ckpt['eta_max'] - ckpt['eta_min'] + 1e-8)
    eta_t = torch.tensor(eta_scaled, dtype=torch.float32, device=device)

    samples = cvae.sample(eta_t, n_samples).detach().cpu().numpy()
    x_gen = samples[:, 0].astype(np.float32)
    m_gen = samples[:, 1].astype(np.float32) if samples.shape[1] >= 2 else None

    print(f'Loaded CVAE samples from {os.path.basename(save_path)}')
    print(f'use_bn    : {ckpt.get("use_bn", False)}')
    print(f'n_samples : {len(x_gen)}')
    return x_gen, m_gen, ckpt


def show_cvae_sample_results(
    name,
    save_path,
    eta_path,
    chunk_dir,
    num_chunks,
    barr_type='barr',
    bins=80,
    pattern='*.h5',
    row_batch_size=50_000,
    device=None,
):
    x, m, ckpt = load_cvae_xt_mt_from_chunks(
        save_path=save_path,
        eta_path=eta_path,
        chunk_dir=chunk_dir,
        num_chunks=num_chunks,
        pattern=pattern,
        row_batch_size=row_batch_size,
        barr_type=barr_type,
        device=device,
    )

    x_stats = statics_result(f'{name} X_T', x)
    m_stats = None
    if barr_type == 'barr' and _as_nonempty_array(m) is not None:
        m_stats = statics_result(f'{name} M_T', m)

    plot_1d(f'{name} X_T', x, bins=bins)
    if barr_type == 'barr' and _as_nonempty_array(m) is not None:
        plot_1d(f'{name} M_T', m, bins=bins)
        plot_2d_density(
            x,
            m,
            title=name,
            xlabel='X_T',
            ylabel='M_T',
            bins=bins,
        )

    return {
        'x': x,
        'm': m,
        'ckpt': ckpt,
        'x_stats': x_stats,
        'm_stats': m_stats,
    }


# =================
# price helpers
# =================
def vanilla_price_from_xt(x, K, r, T, opt_type='call'):
    x = np.asarray(x)
    s_t = np.exp(x)
    if opt_type == 'call':
        payoff = np.maximum(s_t - K, 0.0)
    elif opt_type == 'put':
        payoff = np.maximum(K - s_t, 0.0)
    else:
        raise ValueError("opt_type must be 'call' or 'put'")
    return float(np.exp(-r * T) * np.mean(payoff))


def barrier_price_from_xt_mt(x, m, K, r, T, opt_type='call', B=0.8):
    x = np.asarray(x)
    m = np.asarray(m)
    if len(x) != len(m):
        raise ValueError('x and m must have the same length')

    s_t = np.exp(x)
    alive = m > np.log(B)
    if opt_type == 'call':
        payoff = np.maximum(s_t - K, 0.0) * alive
    elif opt_type == 'put':
        payoff = np.maximum(K - s_t, 0.0) * alive
    else:
        raise ValueError("opt_type must be 'call' or 'put'")
    return float(np.exp(-r * T) * np.mean(payoff))


def price_error(price, ref_price):
    return (price - ref_price) / (ref_price + 1e-12) * 100.0
