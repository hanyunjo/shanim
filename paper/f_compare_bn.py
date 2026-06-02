import os
import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from e_2_CVAE import CVAE

# =================
# load data
# =================
def load_real_xt_mt_from_chunks(chunk_dir, num_chunks):
    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, '*.h5')))

    if len(chunk_paths) == 0:
        raise FileNotFoundError(f'No chunk files found in {chunk_dir} with pattern={pattern}')
    if num_chunks < 1:
        raise ValueError('num_chunks must be >= 1')
    if num_chunks > len(chunk_paths):
        raise ValueError(f'num_chunks={num_chunks} is larger than available chunks={len(chunk_paths)}')

    selected_paths = chunk_paths[:num_chunks]
    x_list = []
    m_list = []
    has_m = None

    for chunk_path in selected_paths:
        with h5py.File(chunk_path, 'r') as f:
            paths = f['paths'][:]

        x_list.append(paths[:, 1].astype(np.float32))
        x_real = np.concatenate(x_list)

        if has_m is None:
            has_m = paths.shape[1] >= 3
        if has_m:
            m_list.append(paths[:, 2].astype(np.float32))
            m_real = np.concatenate(m_list)

    print(f'Loaded real data from {num_chunks}/{len(chunk_paths)} chunks')
    print(f'first chunk: {os.path.basename(selected_paths[0])}')
    print(f'last chunk : {os.path.basename(selected_paths[-1])}')
    print(f'n_real     : {len(x_real)}')

    return x_real, m_real


@torch.no_grad()
def load_cvae_xt_mt(save_path, test_etas, n_samples=100000):
    if not torch.cuda.is_available():
        raise ValueError('Cannot use GPU cuda')

    device = torch.device('cuda')
    ckpt = torch.load(save_path, map_location=device, weights_only=False)

    cvae = CVAE(
        dim_x=ckpt['dim_x'],
        dim_eta=ckpt['dim_eta'],
        dim_z=ckpt['dim_z'],
        hidden_dims=ckpt['hidden_dims'],
        use_bn=ckpt.get('use_bn', False),
    ).to(device)

    cvae.load_state_dict(ckpt['model_state'])
    cvae.eval()

    eta_raw = np.array(test_etas, dtype=np.float32)
    eta_scaled = (eta_raw - ckpt['eta_min']) / (ckpt['eta_max'] - ckpt['eta_min'] + 1e-8)
    eta_t = torch.tensor(eta_scaled, dtype=torch.float32, device=device)

    samples = cvae.sample(eta_t, n_samples).detach().cpu().numpy()
    x_gen = samples[:, 0]
    m_gen = samples[:, 1] if samples.shape[1] >= 2 else None

    print(f'Loaded CVAE samples from {os.path.basename(save_path)}')
    print(f'use_bn     : {ckpt.get("use_bn", False)}')
    print(f'n_samples  : {len(x_gen)}')

    return x_gen, m_gen, ckpt


# =================
# print result
# =================
def statics_result(name, x):
    x = np.asarray(x)
    print()
    print(name)
    print(f'mean : {x.mean():.6f}')
    print(f'std  : {x.std():.6f}')
    print(f'var  : {x.var():.6f}')


def plot_1d(name, x, bins=80):
    x = np.asarray(x)
    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=bins, density=True, alpha=0.7)
    plt.axvline(x.mean(), color='red', linestyle='--', linewidth=1.5, label='mean')
    plt.xlabel(name)
    plt.ylabel('Density')
    plt.title(f'{name} distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_2d_density(x, y, title='2D density', xlabel='X_T', ylabel='M_T', bins=120, cmap='viridis'):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError('x and y must have the same length')

    plt.figure(figsize=(5.5, 5))
    plt.hist2d(x, y, bins=bins, density=True, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='Density')
    plt.tight_layout()
    plt.show()