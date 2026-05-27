"""
데이터 구조:
  {model}_eta.h5              → etas:  (N, dim_eta)
  {model}_chunks/
    {model}_chunk_000.h5     → paths: (chunk_size, 3) = [ori_idx, X_T, M_T]
"""

import numpy as np
import torch
import h5py
import glob
import os
from torch.utils.data import Dataset, DataLoader
from e_2_CVAE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs_CHUNK_DIR  = "/mnt/d/bs_chunks/"
bs_ETA_PATH   = "/mnt/d/bs_eta.h5"
hes_CHUNK_DIR = "/mnt/d/heston_chunks/"
hes_ETA_PATH  = "/mnt/d/heston_eta.h5"

# ──────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────
class ChunkDataset(Dataset):
    # 청크 파일 하나를 RAM에 로드.
    def __init__(self, chunk_path, etas, eta_min, eta_max, model_type='hes'):

        with h5py.File(chunk_path, 'r') as f:
            paths = f['paths'][:] # (N, 3) : [ori_idx, X_T, M_T]

        ori_idx = paths[:, 0].astype(int)
        X_T     = paths[:, 1].astype(np.float32)
        M_T     = paths[:, 2].astype(np.float32)

        eta_matched = etas[ori_idx].astype(np.float32)
        eta_matched = (eta_matched - eta_min) / (eta_max - eta_min + 1e-8)

        if model_type == 'hes':
            self.x = torch.tensor(np.stack([X_T, M_T], axis=1))  # (N, 2)
        else:  # bs
            self.x = torch.tensor(np.stack([X_T], axis=1))       # (N, 1)

        self.eta = torch.tensor(eta_matched)   # (N, dim_eta)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.eta[idx]


def compute_eta_stats(eta_path):
    print("eta min/max 계산 중")
    with h5py.File(eta_path, 'r') as f:
        etas = f['etas'][:].astype(np.float32)   # (M, dim_eta)

    eta_min = etas.min(axis=0)
    eta_max = etas.max(axis=0)
    print("계산 완료\n")
    return etas, eta_min, eta_max


# ─────────────
# 2.train
# ─────────────
def train_chunk(model_type='hes', dim_z=8, hidden_dims=None, batch_size=1024,
                n_epochs=200, lr=1e-3, beta=1.0, save_path='cvae.pt'):
    
    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    if model_type == 'hes':
        chunk_dir = hes_CHUNK_DIR
        eta_path  = hes_ETA_PATH
        dim_x     = 2   # (X_T, M_T)
    else:  # bs
        chunk_dir = bs_CHUNK_DIR
        eta_path  = bs_ETA_PATH
        dim_x     = 1   # (X_T)

    chunk_paths = glob.glob(os.path.join(chunk_dir, "*.h5"))
    n_chunks    = len(chunk_paths)
    etas, eta_min, eta_max = compute_eta_stats(eta_path)

    dim_eta = etas.shape[1]

    # 모델 생성
    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims)

    # 기존 모델 확인
    if os.path.exists(save_path):
        choice = input(f"{save_path} 존재합니다. (1:불러오기 / 2:덮어쓰기 / 3:중지): ")

        if choice == '1':
            load_model = torch.load(save_path)
            cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta,
                        dim_z=load_model['dim_z'], hidden_dims=load_model['hidden_dims'])
            cvae.load_state_dict(load_model['model_state'])
            print("모델 불러오기 완료")
            return cvae, load_model['loss_history'], eta_min, eta_max

        elif choice == '2':
            print("덮어쓰기 학습")
        elif choice == '3':
            raise ValueError("중지")

    # 학습
    cvae.to(device)
    optimizer    = torch.optim.Adam(cvae.parameters(), lr=lr)
    scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}

    for epoch in range(1, n_epochs + 1):
        epoch_recon = 0.0
        epoch_kl    = 0.0
        n_batches   = 0

        chunk_order = np.random.permutation(n_chunks)   # 청크 순서 셔플

        for ci in chunk_order:
            dataset    = ChunkDataset(chunk_paths[ci], etas, eta_min, eta_max, model_type)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True)

            cvae.train()
            for x_batch, eta_batch in dataloader:
                x_batch   = x_batch.to(device)
                eta_batch = eta_batch.to(device)

                recon_loss, kl_loss = cvae(x_batch, eta_batch)
                loss = recon_loss + beta * kl_loss

                optimizer.zero_grad()
                loss.backward() # gradient calcul
                torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0) # gradient explosion correction
                optimizer.step() # weight update

                epoch_recon += recon_loss.item()
                epoch_kl    += kl_loss.item()
                n_batches   += 1

            del dataset, dataloader   # 청크 메모리 해제

        scheduler.step()

        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches
        avg_total = avg_recon + beta * avg_kl

        loss_history['recon_loss'].append(avg_recon)
        loss_history['KL_loss'].append(avg_kl)
        loss_history['total_loss'].append(avg_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_total:.4f}")

        # 10 epoch마다 중간 저장
        if epoch % 10 == 0:
            torch.save({
                'model_state' : cvae.state_dict(),
                'eta_min'     : eta_min,
                'eta_max'     : eta_max,
                'dim_x'       : dim_x,
                'dim_eta'     : dim_eta,
                'dim_z'       : dim_z,
                'hidden_dims' : hidden_dims,
                'loss_history': loss_history,
            }, save_path)
            print(f"  중간 저장 완료 (epoch {epoch})")

    torch.save({
        'model_state' : cvae.state_dict(),
        'eta_min'     : eta_min,
        'eta_max'     : eta_max,
        'dim_x'       : dim_x,
        'dim_eta'     : dim_eta,
        'dim_z'       : dim_z,
        'hidden_dims' : hidden_dims,
        'loss_history': loss_history,
    }, save_path)
    print(f"\n모델 저장 완료: {save_path}")

    return cvae, loss_history, eta_min, eta_max


# ─────────────
# 3. 가격 출력
# ─────────────
def compare_prices(cvae, B, K, eta_min, eta_max, etas, eta_keys, bench_price,
                   opt_type='call', barr_type='barr', n_samples=10000):

    print("\n" + "="*70)
    header = " | ".join(f"{k:>6}" for k in eta_keys)
    print(f"{header} | {'CVAE':>8} | {'bench':>8} | {'오차%':>7}")
    print("="*70)

    for eta in etas:
        eta      = np.array(eta, dtype=np.float32)
        r        = eta[0]
        T        = eta[-1]
        eta_norm = (eta - eta_min) / (eta_max - eta_min + 1e-8)
        eta_t    = torch.tensor(eta_norm, dtype=torch.float32).to(device)


        if barr_type == 'barr':
            cvae_price = cvae.price_barrier(eta_t, B, K, float(r), float(T),
                                            opt_type, n_samples)
        else:
            cvae_price = cvae.price_vanilla(eta_t, K, float(r), float(T),
                                            opt_type, n_samples)

        err      = (cvae_price - bench_price) / (bench_price + 1e-10) * 100
        eta_vals = " | ".join(f"{v:>6.3f}" for v in eta)
        print(f"{eta_vals} | {cvae_price:>8.7f} | {bench_price:>8.7f} | {err:>+6.2f}%")

    print("="*70)
