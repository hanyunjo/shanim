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
    def __init__(self, chunk_path, etas, eta_min, eta_max):

        with h5py.File(chunk_path, 'r') as f:
            paths = f['paths'][:] # (N, 3) : [ori_idx, X_T, M_T]

        ori_idx = paths[:, 0].astype(int)
        X_T     = paths[:, 1].astype(np.float32)
        M_T     = paths[:, 2].astype(np.float32)

        eta_matched = etas[ori_idx].astype(np.float32)
        eta_matched = (eta_matched - eta_min) / (eta_max - eta_min + 1e-8)

        self.x = torch.tensor(np.stack([X_T, M_T], axis=1))  # (N, 2)

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
def train_chunk(model_type = 'hes', dim_z=8, hidden_dims=None, batch_size=1024,
                lr=1e-3, beta=1.0, warmup_chunks=None, use_bn=False, bn_chunks=None, 
                num_chunks=None, shuffle_chunks=True, save_path=None, resume_path=None):
    
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
        dataset = ChunkDataset(chunk_paths[ci], etas, eta_min, eta_max)
        return dataset

    def apply_bn_mode(global_chunk):
        if not use_bn:
            return "off"
        if bn_chunks is not None and global_chunk >= bn_chunks:
            freeze_batchnorm(cvae)
            return "frozen"
        cvae.train()
        return "train"

    def beta_eff_for_chunk(global_chunk):
        if warmup_chunks is None or warmup_chunks == 0:
            return float(beta)
        ratio = min(1.0, float(global_chunk + 1) / float(warmup_chunks))
        return float(beta) * ratio

    def train_one_chunk(dataset, global_chunk):
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
        bn_mode = apply_bn_mode(global_chunk)
        beta_eff = beta_eff_for_chunk(global_chunk)
        chunk_recon = 0.0
        chunk_kl = 0.0
        chunk_total = 0.0
        chunk_batches = 0
        for x_batch, eta_batch in dataloader:
            x_batch   = x_batch.to(device,non_blocking=True) # CPU에서 GPU로 비동기식 전송
            eta_batch = eta_batch.to(device, non_blocking=True)

            recon_loss, kl_loss = cvae(x_batch, eta_batch)
            loss = recon_loss + beta_eff * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
            optimizer.step()

            recon_value = recon_loss.item()
            kl_value = kl_loss.item()
            total_value = loss.item()
            epoch_accum['recon_sum'] += recon_value
            epoch_accum['kl_sum'] += kl_value
            epoch_accum['total_sum'] += total_value
            epoch_accum['n_batches'] += 1
            chunk_recon += recon_value
            chunk_kl += kl_value
            chunk_total += total_value
            chunk_batches += 1

        del dataloader

        chunk_batches = max(chunk_batches, 1)
        chunk_avg_recon = chunk_recon / chunk_batches
        chunk_avg_kl = chunk_kl / chunk_batches
        chunk_avg_total = chunk_total / chunk_batches
        return chunk_avg_recon, chunk_avg_kl, chunk_avg_total, bn_mode, beta_eff

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
            'bn_chunks'   : bn_chunks,
            'warmup_chunks': warmup_chunks,
            'loss_history': loss_history, # epoch별 평균 손실 기록
            'chunk_loss_history': chunk_loss_history,
            'num_chunks'  : num_chunks,
            'trained_chunks': current_chunks,
            'total_chunks': total_chunks,
            'shuffle_chunks': shuffle_chunks,
            'epoch'          : completed_epochs,
            'epoch_accum'    : epoch_accum,
            'optimizer_state': optimizer.state_dict(),
        }, save_path)

    def record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses):
        chunk_avg_recon, chunk_avg_kl, chunk_avg_total, bn_mode, beta_eff = chunk_losses
        chunk_loss_history.setdefault('beta_eff', [])
        chunk_loss_history['epoch'].append(epoch)
        chunk_loss_history['global_chunk'].append(global_chunk)
        chunk_loss_history['chunk_pos'].append(chunk_pos) # 학습 chunk 개수
        chunk_loss_history['chunk_idx'].append(int(ci)) # 학습 chunk 파일 인덱스
        chunk_loss_history['chunk_file'].append(os.path.basename(chunk_paths[int(ci)]))
        chunk_loss_history['recon_loss'].append(chunk_avg_recon)
        chunk_loss_history['KL_loss'].append(chunk_avg_kl)
        chunk_loss_history['total_loss'].append(chunk_avg_total)
        chunk_loss_history['beta_eff'].append(beta_eff)
        print(
            f"Chunk step {global_chunk + 1:5d} | "
            f"epoch {epoch:4d} chunk {chunk_pos + 1:3d}/{total_chunks} | "
            f"file_idx {ci:3d} | "
            f"BN {bn_mode:6s} | beta_eff: {beta_eff:.4f} | "
            f"Recon: {chunk_avg_recon:.4f} | KL: {chunk_avg_kl:.4f} | Total: {chunk_avg_total:.4f}",
            flush=True,
        )

    def finish_epoch(epoch, epoch_start, current_chunks):
        n_batches = max(int(epoch_accum['n_batches']), 1)
        avg_recon = epoch_accum['recon_sum'] / n_batches
        avg_kl = epoch_accum['kl_sum'] / n_batches
        avg_total = epoch_accum['total_sum'] / n_batches

        loss_history['recon_loss'].append(avg_recon)
        loss_history['KL_loss'].append(avg_kl)
        loss_history['total_loss'].append(avg_total)

        epoch_time = time.perf_counter() - epoch_start
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Epoch {epoch:4d} 완료 |\n"
              f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_total:.4f} |\n"
              f"epoch time : {epoch_time/60:.2f}m |\n"
              f"GPU mem: {gpu_mem:.2f}GB"
              )

        epoch_accum['recon_sum'] = 0.0
        epoch_accum['kl_sum'] = 0.0
        epoch_accum['total_sum'] = 0.0
        epoch_accum['n_batches'] = 0

        if epoch % 10 == 0:
            save_checkpoint(current_chunks)
            print(f"  중간 저장 완료 (epoch {epoch})")

    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    if model_type == 'hes':
        chunk_dir = HES_CHUNK_DIR
        eta_path  = HES_ETA_PATH
    elif model_type == 'bs':
        chunk_dir = BS_CHUNK_DIR
        eta_path  = BS_ETA_PATH
    else:
        raise ValueError("model_type must be 'hes' or 'bs'")

    dim_x = 2   # (X_T, M_T)

    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.h5")), key=_chunk_sort_key)
    total_chunks = len(chunk_paths)
    if total_chunks == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    if num_chunks is None:
        raise ValueError("Input num_chunks")
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if bn_chunks is not None:
        bn_chunks = int(bn_chunks)
        if bn_chunks < 0:
            raise ValueError("bn_chunks must be >= 0")
    if warmup_chunks is not None:
        warmup_chunks = int(warmup_chunks)
        if warmup_chunks < 0:
            raise ValueError("warmup_chunks must be >= 0")
    if save_path is None:
        raise ValueError("Input save_path")
    if os.path.exists(save_path) and resume_path is None:
        choice = input(f"{save_path} 존재합니다. (1:덮어쓰기 / 2:중지): ")
        if choice == '1':
            print("덮어쓰기 학습")
        elif choice == '2':
            raise ValueError("중지")

    etas, eta_min, eta_max = compute_eta_stats(eta_path)
    dim_eta = etas.shape[1]

    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        checkpoint_use_bn = bool(resume_checkpoint.get('use_bn', False))
        if checkpoint_use_bn != bool(use_bn):
            print(f"use_bn을 checkpoint 설정({checkpoint_use_bn})으로 맞춥니다.")
        use_bn = checkpoint_use_bn

        checkpoint_hidden_dims = resume_checkpoint.get('hidden_dims', hidden_dims)
        if checkpoint_hidden_dims != hidden_dims:
            print(f"hidden_dims를 checkpoint 설정({checkpoint_hidden_dims})으로 맞춥니다.")
        hidden_dims = checkpoint_hidden_dims

        checkpoint_dim_z = int(resume_checkpoint.get('dim_z', dim_z))
        if checkpoint_dim_z != dim_z:
            print(f"dim_z를 checkpoint 설정({checkpoint_dim_z})으로 맞춥니다.")
        dim_z = checkpoint_dim_z

        checkpoint_dim_x = int(resume_checkpoint.get('dim_x', dim_x))
        if checkpoint_dim_x != dim_x:
            raise ValueError(f"checkpoint dim_x={checkpoint_dim_x}, current dim_x={dim_x}")

        checkpoint_dim_eta = int(resume_checkpoint.get('dim_eta', dim_eta))
        if checkpoint_dim_eta != dim_eta:
            raise ValueError(f"checkpoint dim_eta={checkpoint_dim_eta}, current eta dim={dim_eta}")

    # 모델 생성
    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims, use_bn=use_bn)
    cvae.bn_chunks = bn_chunks
    cvae.to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    completed_chunks = 0
    epoch_accum = {'recon_sum': 0.0, 'kl_sum': 0.0, 'total_sum': 0.0, 'n_batches': 0}
    empty_chunk_history = {
        'epoch': [], 'global_chunk': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [],
        'recon_loss': [], 'KL_loss': [], 'total_loss': [], 'beta_eff': []
    }

    if resume_path is not None: # load resume model
        cvae.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        loss_history = resume_checkpoint["loss_history"]
        chunk_loss_history = resume_checkpoint.get('chunk_loss_history', empty_chunk_history.copy())
        chunk_loss_history.setdefault('global_chunk', [])
        if len(chunk_loss_history['global_chunk']) < len(chunk_loss_history.get('chunk_idx', [])):
            chunk_loss_history['global_chunk'] = list(range(len(chunk_loss_history.get('chunk_idx', []))))
        eta_min = resume_checkpoint["eta_min"]
        eta_max = resume_checkpoint["eta_max"]
        completed_chunks = int(resume_checkpoint.get("trained_chunks", len(chunk_loss_history.get('chunk_idx', [])))) # 전에 실행된 chunk파일 
        checkpoint_bn_chunks = resume_checkpoint.get('bn_chunks', None)
        checkpoint_warmup_chunks = resume_checkpoint.get('warmup_chunks', warmup_chunks)
        if checkpoint_bn_chunks is None:
            if use_bn and bn_chunks is not None and completed_chunks > bn_chunks:
                raise ValueError(
                    f"이전 checkpoint에는 bn_chunks가 없고 이미 {completed_chunks} chunks가 BN으로 학습됐습니다. "
                    f"bn_chunks={bn_chunks}로는 과거 BN 적용을 되돌릴 수 없어 resume할 수 없습니다."
                )
            if bn_chunks is not None:
                print(f"기존 checkpoint에 bn_chunks가 없어 새 설정({bn_chunks})으로 이어서 저장합니다.")
        elif checkpoint_bn_chunks != bn_chunks:
            raise ValueError(
                f"bn_chunks가 checkpoint 설정({checkpoint_bn_chunks})과 다릅니다. "
                f"resume 입력값: {bn_chunks}"
            )
        if checkpoint_warmup_chunks != warmup_chunks:
            raise ValueError(
                f"warmup_chunks가 checkpoint 설정({checkpoint_warmup_chunks})과 다릅니다. "
                f"resume 입력값: {warmup_chunks}"
            )
        epoch_accum = resume_checkpoint.get('epoch_accum', epoch_accum)
        epoch_accum.setdefault('total_sum', 0.0)
        print(f"체크포인트 재개: {resume_path} | 완료 chunks={completed_chunks}")
    else:
        loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}
        chunk_loss_history = empty_chunk_history.copy()

    # start train
    train_start = time.perf_counter()
    target_chunks = completed_chunks + num_chunks
    print(
        f"학습 시작 | 이번 실행 chunks={num_chunks} | "
        f"진행 chunks={completed_chunks}->{target_chunks} | files/epoch={total_chunks} | "
        f"bn_chunks={bn_chunks} | warmup_chunks={warmup_chunks}"
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

            chunk_losses = train_one_chunk(dataset, global_chunk)
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

    dim_x = 2

    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.h5")))
    if len(chunk_paths) == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    etas, eta_min, eta_max = compute_eta_stats(eta_path)
    dim_eta = etas.shape[1]

    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    # chunk load time
    load_start = time.perf_counter()
    dataset = ChunkDataset(chunk_paths[chunk_idx], etas, eta_min, eta_max)
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