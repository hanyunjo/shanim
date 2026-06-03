import time
import numpy as np
import torch
import h5py
import glob
import os
import re
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from e_2_CVAE import *

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")

# _basic(eta에 B가 없는 버전), _B
BS_CHUNK_DIR  = "/mnt/d/bs_chunks_correction/"
BS_ETA_PATH   = "/mnt/d/bs_eta_basic.h5"
HES_CHUNK_DIR = "/mnt/d/heston_chunks_correction/"
HES_ETA_PATH  = "/mnt/d/heston_eta_basic.h5"


# ──────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────
class ChunkDataset(Dataset):
    # 청크 파일 하나를 RAM에 로드.
    def __init__(self, chunk_path, etas, eta_min, eta_max, barr_type='barr'):

        with h5py.File(chunk_path, 'r') as f:
            paths = f['paths'][:] # (N, 3) : [ori_idx, X_T, M_T]

        ori_idx = paths[:, 0].astype(int)
        X_T     = paths[:, 1].astype(np.float32)
        M_T     = paths[:, 2].astype(np.float32)

        eta_matched = etas[ori_idx].astype(np.float32)
        eta_matched = (eta_matched - eta_min) / (eta_max - eta_min + 1e-8)

        if barr_type == 'barr':
            self.x = torch.tensor(np.stack([X_T, M_T], axis=1))  # (N, 2)
        else:  # van
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



def _chunk_sort_key(chunk_path):
    numbers = re.findall(r"\d+", os.path.basename(chunk_path))
    return int(numbers[-1]) if numbers else -1


# ─────────────
# 2.train
# ─────────────
def train_chunk(model_type = 'hes', barr_type='barr', dim_z=8, hidden_dims=None, batch_size=1024,
                lr=1e-3, beta=1.0, use_bn=False, num_chunks=None,
                shuffle_chunks=True, save_path='cvae.pt', resume_path=None):
    
    def chunk_order_for_epoch(epoch):
        if shuffle_chunks:
            rng = np.random.default_rng(epoch)
            return rng.permutation(total_chunks)
        return np.arange(total_chunks)

    def chunk_info(global_chunk):
        epoch = global_chunk // total_chunks + 1
        chunk_pos = global_chunk % total_chunks
        chunk_order = chunk_order_for_epoch(epoch)
        ci = int(chunk_order[chunk_pos])
        return epoch, chunk_pos, ci

    def load_chunk(ci):
        dataset = ChunkDataset(chunk_paths[ci], etas, eta_min, eta_max, barr_type)
        return dataset

    def train_one_chunk(dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True, # GPU가 빠르게 가져갈 수 있는 CPU 메모리 영역에 올려둠.
            drop_last=True
        )

        cvae.train()
        chunk_recon = 0.0
        chunk_kl = 0.0
        chunk_batches = 0
        for x_batch, eta_batch in dataloader:
            x_batch   = x_batch.to(device,non_blocking=True) # CPU에서 GPU로 비동기식 전송
            eta_batch = eta_batch.to(device, non_blocking=True)

            recon_loss, kl_loss = cvae(x_batch, eta_batch)
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
            optimizer.step()

            recon_value = recon_loss.item()
            kl_value = kl_loss.item()
            epoch_accum['recon_sum'] += recon_value
            epoch_accum['kl_sum'] += kl_value
            epoch_accum['n_batches'] += 1
            chunk_recon += recon_value
            chunk_kl += kl_value
            chunk_batches += 1

        del dataloader

        chunk_batches = max(chunk_batches, 1)
        chunk_avg_recon = chunk_recon / chunk_batches
        chunk_avg_kl = chunk_kl / chunk_batches
        chunk_avg_total = chunk_avg_recon + beta * chunk_avg_kl
        return chunk_avg_recon, chunk_avg_kl, chunk_avg_total

    def save_checkpoint(current_chunks):
        completed_epochs = current_chunks // total_chunks
        torch.save({
            'model_state' : cvae.state_dict(),
            'eta_min'     : eta_min,
            'eta_max'     : eta_max,
            'dim_x'       : dim_x,
            'dim_eta'     : dim_eta,
            'dim_z'       : dim_z,
            'hidden_dims' : hidden_dims,
            'use_bn'      : use_bn,
            'loss_history': loss_history, # epoch별 평균 손실 기록
            'chunk_loss_history': chunk_loss_history,
            'num_chunks'  : num_chunks,
            'trained_chunks': current_chunks,
            'total_chunks': total_chunks,
            'shuffle_chunks': shuffle_chunks,
            'epoch'          : completed_epochs,
            'epoch_accum'    : epoch_accum,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }, save_path)

    def record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses):
        chunk_avg_recon, chunk_avg_kl, chunk_avg_total = chunk_losses
        chunk_loss_history['epoch'].append(epoch)
        chunk_loss_history['global_chunk'].append(global_chunk)
        chunk_loss_history['chunk_pos'].append(chunk_pos) # 학습 chunk 개수
        chunk_loss_history['chunk_idx'].append(int(ci)) # 학습 chunk 파일 인덱스
        chunk_loss_history['chunk_file'].append(os.path.basename(chunk_paths[int(ci)]))
        chunk_loss_history['recon_loss'].append(chunk_avg_recon)
        chunk_loss_history['KL_loss'].append(chunk_avg_kl)
        chunk_loss_history['total_loss'].append(chunk_avg_total)
        print(
            f"Chunk step {global_chunk + 1:5d} | "
            f"epoch {epoch:4d} chunk {chunk_pos + 1:3d}/{total_chunks} | "
            f"file_idx {ci:3d} | "
            f"Recon: {chunk_avg_recon:.4f} | KL: {chunk_avg_kl:.4f} | Total: {chunk_avg_total:.4f}",
            flush=True,
        )

    def finish_epoch(epoch, epoch_start, current_chunks):
        n_batches = max(int(epoch_accum['n_batches']), 1)
        avg_recon = epoch_accum['recon_sum'] / n_batches
        avg_kl = epoch_accum['kl_sum'] / n_batches
        avg_total = avg_recon + beta * avg_kl

        loss_history['recon_loss'].append(avg_recon)
        loss_history['KL_loss'].append(avg_kl)
        loss_history['total_loss'].append(avg_total)

        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Epoch {epoch:4d} 완료 |\n"
              f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_total:.4f} |\n"
              f"epoch time : {epoch_time/60:.2f}m |\n"
              f"GPU mem: {gpu_mem:.2f}GB"
              )

        epoch_accum['recon_sum'] = 0.0
        epoch_accum['kl_sum'] = 0.0
        epoch_accum['n_batches'] = 0

        if epoch % 10 == 0:
            save_checkpoint(current_chunks)
            print(f"  중간 저장 완료 (epoch {epoch})")


    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    if model_type == 'hes':
        chunk_dir = HES_CHUNK_DIR
        eta_path  = HES_ETA_PATH
    else:  # bs
        chunk_dir = BS_CHUNK_DIR
        eta_path  = BS_ETA_PATH

    if barr_type == 'barr':
        dim_x     = 2   # (X_T, M_T)
    else: # van
        dim_x     = 1   # (X_T)

    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.h5")), key=_chunk_sort_key)
    total_chunks = len(chunk_paths)
    if total_chunks == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    if num_chunks is None:
        num_chunks = total_chunks
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")

    save_path = os.path.expanduser(save_path)
    resume_path = os.path.expanduser(resume_path) if resume_path is not None else None

    # 기존 모델 확인
    if resume_path is None and os.path.exists(save_path):
        choice = input(f"{save_path} 존재합니다. (1:불러오기 / 2:덮어쓰기 / 3:중지): ")

        if choice == '1':
            existing_model = torch.load(save_path, map_location=device, weights_only=False)
            cvae = CVAE(dim_x=existing_model['dim_x'], 
                        dim_eta=existing_model['dim_eta'],
                        dim_z=existing_model['dim_z'], 
                        hidden_dims=existing_model['hidden_dims'],
                        use_bn=existing_model.get('use_bn', False)
                        )
            cvae.load_state_dict(existing_model['model_state'])
            cvae.to(device)
            cvae.eval()
            print("모델 불러오기 완료")
            return cvae, existing_model['loss_history'], existing_model['eta_min'], existing_model['eta_max']

        elif choice == '2':
            print("덮어쓰기 학습")
        elif choice == '3':
            raise ValueError("중지")

    etas, eta_min, eta_max = compute_eta_stats(eta_path)
    dim_eta = etas.shape[1]

    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    checkpoint = resume_checkpoint
    if checkpoint is not None:
        checkpoint_use_bn = bool(checkpoint.get('use_bn', False))
        if checkpoint_use_bn != bool(use_bn):
            print(f"use_bn을 checkpoint 설정({checkpoint_use_bn})으로 맞춥니다.")
        use_bn = checkpoint_use_bn
    else:
        use_bn = bool(use_bn)

    # 모델 생성
    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims, use_bn=use_bn)

    # 학습
    cvae.to(device)
    optimizer    = torch.optim.Adam(cvae.parameters(), lr=lr)
    scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    completed_chunks = 0
    epoch_accum = {'recon_sum': 0.0, 'kl_sum': 0.0, 'n_batches': 0}
    empty_chunk_history = {
        'epoch': [], 'global_chunk': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [],
        'recon_loss': [], 'KL_loss': [], 'total_loss': []
    }
    if resume_path is not None: # chunk 진행량을 이어서 학습
        ckpt = resume_checkpoint
        cvae.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        loss_history = ckpt["loss_history"]
        chunk_loss_history = ckpt.get('chunk_loss_history', empty_chunk_history.copy())
        chunk_loss_history.setdefault('global_chunk', [])
        if len(chunk_loss_history['global_chunk']) < len(chunk_loss_history.get('chunk_idx', [])):
            chunk_loss_history['global_chunk'] = list(range(len(chunk_loss_history.get('chunk_idx', []))))
        eta_min = ckpt["eta_min"]
        eta_max = ckpt["eta_max"]
        completed_chunks = int(ckpt.get("trained_chunks", len(chunk_loss_history.get('chunk_idx', [])))) # 전에 실행된 chunk파일 
        epoch_accum = ckpt.get('epoch_accum', epoch_accum)
        print(f"체크포인트 재개: {resume_path} | 완료 chunks={completed_chunks}")
    else:
        loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}
        chunk_loss_history = empty_chunk_history.copy()

    train_start = time.perf_counter()
    run_start_chunk = completed_chunks
    target_chunks = completed_chunks + num_chunks
    print(
        f"학습 시작 | 이번 실행 chunks={num_chunks} | "
        f"진행 chunks={run_start_chunk}->{target_chunks} | files/epoch={total_chunks}"
    )

    

    current_epoch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        _, _, first_ci = chunk_info(completed_chunks)
        future = prefetcher.submit(load_chunk, first_ci)

        for offset in range(num_chunks):
            global_chunk = completed_chunks + offset
            epoch, chunk_pos, ci = chunk_info(global_chunk)
            if chunk_pos == 0:
                current_epoch_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

            dataset = future.result()
            if offset + 1 < num_chunks:
                _, _, next_ci = chunk_info(global_chunk + 1)
                future = prefetcher.submit(load_chunk, next_ci)

            chunk_losses = train_one_chunk(dataset)
            record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses)

            del dataset

            current_chunks = global_chunk + 1
            if current_chunks % total_chunks == 0:
                finish_epoch(epoch, current_epoch_start, current_chunks)

    total_time = time.perf_counter() - train_start
    save_checkpoint(target_chunks)
    print(f"모델 저장 완료: {save_path}")
    print(f"총 학습 시간: {total_time/60:.2f}분 ({total_time/3600:.2f}시간)")

    return cvae, loss_history, eta_min, eta_max

# ─────────────
# 2-1. test one chunk train time
# ─────────────
def benchmark_one_chunk(model_type='hes', barr_type='barr', dim_z=8, hidden_dims=None,
                        batch_size=1024, lr=1e-3, beta=1.0, chunk_idx=0):
    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    if model_type == 'hes':
        chunk_dir = HES_CHUNK_DIR
        eta_path  = HES_ETA_PATH
    else:
        chunk_dir = BS_CHUNK_DIR
        eta_path  = BS_ETA_PATH

    dim_x = 2 if barr_type == 'barr' else 1

    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.h5")))
    if len(chunk_paths) == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    etas, eta_min, eta_max = compute_eta_stats(eta_path)
    dim_eta = etas.shape[1]

    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    # chunk load time
    load_start = time.perf_counter()
    dataset = ChunkDataset(chunk_paths[chunk_idx], etas, eta_min, eta_max, barr_type)
    load_time = time.perf_counter() - load_start

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True, 
        prefetch_factor=2,
    )

    # 정확한 GPU 시간 측정을 위해 synchronize 사용
    torch.cuda.synchronize()
    train_start = time.perf_counter()

    cvae.train()
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0
    print("학습 시작")
    for x_batch, eta_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        eta_batch = eta_batch.to(device, non_blocking=True)

        recon_loss, kl_loss = cvae(x_batch, eta_batch)
        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
        optimizer.step()

        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start

    print(f"chunk file      : {chunk_paths[chunk_idx]}")
    print(f"load time       : {load_time:.2f} sec")
    print(f"train time      : {train_time:.2f} sec")
    print(f"total time      : {load_time + train_time:.2f} sec")
    print(f"n_batches       : {n_batches}")
    print(f"avg batch time  : {train_time / n_batches:.4f} sec")
    print(f"recon loss      : {total_recon / n_batches:.6f}")
    print(f"KL loss         : {total_kl / n_batches:.6f}")
    print(f"GPU memory      : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    return train_time


# ─────────────
# 3. 가격 출력
# ─────────────
def compare_prices(cvae, B, K, eta_min, eta_max, eta, eta_keys, bench_price,
                   opt_type='call', barr_type='barr', n_samples=10000):

    print("\n" + "="*70)
    header = " | ".join(f"{k:>6}" for k in eta_keys)
    print(f"{header} | {'CVAE':>8} | {'bench':>8} | {'오차%':>7}")
    print("="*70)

    eta      = np.array(eta, dtype=np.float32)
    r        = eta[0]
    T        = eta[-1]
    eta_norm = (eta - eta_min) / (eta_max - eta_min + 1e-8)
    eta_t    = torch.tensor(eta_norm, dtype=torch.float32).to(device)

    start = time.time()
    if barr_type == 'barr':
        cvae_price = cvae.price_barrier(eta_t, B, K, float(r), float(T),
                                        opt_type, n_samples)
    else:
        cvae_price = cvae.price_vanilla(eta_t, K, float(r), float(T),
                                        opt_type, n_samples)
    print(f"Pricing time: {time.time() - start:.6f}s")

    err      = (cvae_price - bench_price) / (bench_price + 1e-10) * 100
    eta_vals = " | ".join(f"{v:>6.3f}" for v in eta)
    print(f"{eta_vals} | {cvae_price:>8.7f} | {bench_price:>8.7f} | {err:>+6.2f}%")

    print("="*70)
