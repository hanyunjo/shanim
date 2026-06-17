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


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_time_profile():
    return {
        'chunk_load': 0.0,
        'load_wait': 0.0,
        'dataloader_init': 0.0,
        'batch_fetch': 0.0,
        'h2d': 0.0,
        'forward': 0.0,
        'backward': 0.0,
        'clip': 0.0,
        'step': 0.0,
        'loss_item': 0.0,
        'chunk_total': 0.0,
        'validation': 0.0,
        'batches': 0,
    }


def _print_time_profile(profile):
    batches = max(int(profile.get('batches', 0)), 1)
    print(
        "Time | "
        f"total={profile['chunk_total']:.2f}s | "
        f"chunk_load={profile['chunk_load']:.2f}s | "
        f"load_wait={profile['load_wait']:.2f}s | "
        f"loader_init={profile['dataloader_init']:.2f}s | "
        f"batch_fetch={profile['batch_fetch']:.2f}s ({profile['batch_fetch']/batches:.4f}/batch) | "
        f"h2d={profile['h2d']:.2f}s ({profile['h2d']/batches:.4f}/batch) | "
        f"fwd={profile['forward']:.2f}s ({profile['forward']/batches:.4f}/batch) | "
        f"bwd={profile['backward']:.2f}s ({profile['backward']/batches:.4f}/batch) | "
        f"clip={profile['clip']:.2f}s ({profile['clip']/batches:.4f}/batch) | "
        f"step={profile['step']:.2f}s ({profile['step']/batches:.4f}/batch) | "
        f"batches={profile['batches']}",
        flush=True,
    )

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


def _chunk_file_idx(chunk_path):
    filename = os.path.basename(chunk_path)
    match = re.search(r"chunk_(\d+)", filename)
    if match is None:
        raise ValueError(f"Cannot parse chunk index from filename: {filename}")
    return int(match.group(1))


def _chunk_sort_key(chunk_path):
    return _chunk_file_idx(chunk_path)


# ─────────────
# 2.train
# ─────────────
def train_chunk(model_type = 'hes', dim_z=8, hidden_dims=None, batch_size=1024,
                lr=1e-3, beta=1.0, warmup_chunks=None, use_bn=False, bn_chunks=None, 
                num_chunks=None, shuffle_chunks=True, save_path=None, resume_path=None,
                exclude_chunk_idxs=None, validation_chunk_idxs=None, val_every_chunks=10):
    
    def chunk_order_for_epoch(epoch):
        chunk_indices = np.array(trainable_chunk_idxs, dtype=int)
        if shuffle_chunks:
            rng = np.random.default_rng(epoch)
            return rng.permutation(chunk_indices)
        return chunk_indices

    def chunk_info(global_chunk):
        epoch = global_chunk // total_chunks + 1
        chunk_pos = global_chunk % total_chunks
        chunk_order = chunk_order_for_epoch(epoch)
        ci = int(chunk_order[chunk_pos])
        return epoch, chunk_pos, ci

    def load_chunk(ci):
        t0 = time.perf_counter()
        dataset = ChunkDataset(chunk_path_by_idx[int(ci)], etas, eta_min, eta_max)
        load_time = time.perf_counter() - t0
        return dataset, load_time

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
        profile = _empty_time_profile()
        chunk_t0 = time.perf_counter()

        t0 = time.perf_counter()
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
        profile['dataloader_init'] = time.perf_counter() - t0

        cvae.train()
        bn_mode = apply_bn_mode(global_chunk)
        beta_eff = beta_eff_for_chunk(global_chunk)
        chunk_recon = 0.0
        chunk_kl = 0.0
        chunk_total = 0.0
        chunk_batches = 0

        data_iter = iter(dataloader)
        while True:
            t0 = time.perf_counter()
            try:
                x_batch, eta_batch = next(data_iter)
            except StopIteration:
                break
            profile['batch_fetch'] += time.perf_counter() - t0

            _cuda_sync()
            t0 = time.perf_counter()
            x_batch = x_batch.to(device, non_blocking=True) # CPU에서 GPU로 비동기식 전송
            eta_batch = eta_batch.to(device, non_blocking=True)
            _cuda_sync()
            profile['h2d'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            recon_loss, kl_loss = cvae(x_batch, eta_batch)
            loss = recon_loss + beta_eff * kl_loss
            _cuda_sync()
            profile['forward'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            _cuda_sync()
            profile['backward'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
            _cuda_sync()
            profile['clip'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            optimizer.step()
            _cuda_sync()
            profile['step'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            recon_value = recon_loss.item()
            kl_value = kl_loss.item()
            total_value = loss.item()
            profile['loss_item'] += time.perf_counter() - t0

            epoch_accum['recon_sum'] += recon_value
            epoch_accum['kl_sum'] += kl_value
            epoch_accum['total_sum'] += total_value
            epoch_accum['n_batches'] += 1
            chunk_recon += recon_value
            chunk_kl += kl_value
            chunk_total += total_value
            chunk_batches += 1
            profile['batches'] += 1

        del dataloader

        profile['chunk_total'] = time.perf_counter() - chunk_t0
        chunk_batches = max(chunk_batches, 1)
        chunk_avg_recon = chunk_recon / chunk_batches
        chunk_avg_kl = chunk_kl / chunk_batches
        chunk_avg_total = chunk_total / chunk_batches
        return chunk_avg_recon, chunk_avg_kl, chunk_avg_total, bn_mode, beta_eff, profile

    def validate_chunks(current_chunks):
        if len(validation_chunk_idxs) == 0:
            return None

        was_training = cvae.training
        cvae.eval()
        beta_eff = beta_eff_for_chunk(current_chunks - 1)
        val_recon = 0.0
        val_kl = 0.0
        val_total = 0.0
        val_batches = 0
        val_kl_dim_sum = None

        with torch.no_grad():
            for ci in sorted(validation_chunk_idxs):
                dataset, _ = load_chunk(ci)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    persistent_workers=True,
                    prefetch_factor=2,
                    pin_memory=True,
                    drop_last=False,
                )
                for x_batch, eta_batch in dataloader:
                    x_batch = x_batch.to(device, non_blocking=True)
                    eta_batch = eta_batch.to(device, non_blocking=True)

                    recon_loss, kl_loss, kl_dim_mean = cvae(x_batch, eta_batch, return_kl_dim=True)
                    loss = recon_loss + beta_eff * kl_loss

                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_total += loss.item()
                    val_batches += 1
                    kl_dim_cpu = kl_dim_mean.detach().cpu()
                    if val_kl_dim_sum is None:
                        val_kl_dim_sum = torch.zeros_like(kl_dim_cpu)
                    val_kl_dim_sum += kl_dim_cpu

                del dataloader, dataset

        if was_training:
            cvae.train()

        val_batches = max(val_batches, 1)
        if val_kl_dim_sum is None:
            val_kl_dim_mean = []
        else:
            val_kl_dim_mean = (val_kl_dim_sum / val_batches).tolist()

        return {
            'global_chunk': current_chunks,
            'recon_loss': val_recon / val_batches,
            'KL_loss': val_kl / val_batches,
            'total_loss': val_total / val_batches,
            'beta_eff': beta_eff,
            'kl_dim_mean': val_kl_dim_mean,
        }

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
            'excluded_chunk_idxs': sorted(exclude_chunk_idxs),
            'validation_chunk_idxs': sorted(validation_chunk_idxs),
            'val_every_chunks': val_every_chunks,
            'num_all_chunks': num_all_chunks,
            'all_chunk_idxs': all_chunk_idxs,
            'epoch'          : completed_epochs,
            'epoch_accum'    : epoch_accum,
            'optimizer_state': optimizer.state_dict(),
        }, save_path)

    def record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses):
        chunk_avg_recon, chunk_avg_kl, chunk_avg_total, bn_mode, beta_eff, profile = chunk_losses
        chunk_loss_history.setdefault('beta_eff', [])
        for key in profile:
            chunk_loss_history.setdefault(f'time_{key}', [])
        chunk_loss_history['epoch'].append(epoch)
        chunk_loss_history['global_chunk'].append(global_chunk)
        chunk_loss_history['chunk_pos'].append(chunk_pos) # 학습 chunk 개수
        chunk_loss_history['chunk_idx'].append(int(ci)) # 학습 chunk 파일 인덱스
        chunk_loss_history['chunk_file'].append(os.path.basename(chunk_path_by_idx[int(ci)]))
        chunk_loss_history['recon_loss'].append(chunk_avg_recon)
        chunk_loss_history['KL_loss'].append(chunk_avg_kl)
        chunk_loss_history['total_loss'].append(chunk_avg_total)
        chunk_loss_history['beta_eff'].append(beta_eff)
        for key, value in profile.items():
            chunk_loss_history[f'time_{key}'].append(value)
        print(
            f"Chunk step {global_chunk + 1:5d} | "
            f"epoch {epoch:4d} chunk {chunk_pos + 1:3d}/{total_chunks} | "
            f"file_idx {ci:3d} | "
            f"BN {bn_mode:6s} | beta_eff: {beta_eff:.4f} | "
            f"Recon: {chunk_avg_recon:.4f} | KL: {chunk_avg_kl:.4f} | Total: {chunk_avg_total:.4f}",
            flush=True,
        )
        _print_time_profile(profile)

    def record_validation_loss(val_result):
        if val_result is None:
            return
        chunk_loss_history.setdefault('val_global_chunk', [])
        chunk_loss_history.setdefault('val_recon_loss', [])
        chunk_loss_history.setdefault('val_KL_loss', [])
        chunk_loss_history.setdefault('val_total_loss', [])
        chunk_loss_history.setdefault('val_beta_eff', [])
        chunk_loss_history.setdefault('val_kl_dim_mean', [])
        chunk_loss_history.setdefault('val_chunk_idxs', [])
        chunk_loss_history.setdefault('val_chunk_files', [])

        chunk_loss_history['val_global_chunk'].append(val_result['global_chunk'])
        chunk_loss_history['val_recon_loss'].append(val_result['recon_loss'])
        chunk_loss_history['val_KL_loss'].append(val_result['KL_loss'])
        chunk_loss_history['val_total_loss'].append(val_result['total_loss'])
        chunk_loss_history['val_beta_eff'].append(val_result['beta_eff'])
        chunk_loss_history['val_kl_dim_mean'].append(val_result['kl_dim_mean'])
        chunk_loss_history['val_chunk_idxs'].append(sorted(validation_chunk_idxs))
        chunk_loss_history['val_chunk_files'].append(
            [os.path.basename(chunk_path_by_idx[int(ci)]) for ci in sorted(validation_chunk_idxs)]
        )
        print(
            f"Validation @ chunk {val_result['global_chunk']:5d} | "
            f"Recon: {val_result['recon_loss']:.4f} | KL: {val_result['KL_loss']:.4f} | "
            f"Total: {val_result['total_loss']:.4f} | "
            f"KL_dim: {[round(v, 6) for v in val_result['kl_dim_mean']]}",
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
            t0 = time.perf_counter()
            save_checkpoint(current_chunks)
            checkpoint_time = time.perf_counter() - t0
            print(f"  중간 저장 완료 (epoch {epoch}) | checkpoint_time={checkpoint_time:.2f}s")

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
    num_all_chunks = len(chunk_paths)
    if num_all_chunks == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    all_chunk_idxs = [_chunk_file_idx(path) for path in chunk_paths]
    if len(set(all_chunk_idxs)) != len(all_chunk_idxs):
        raise ValueError(f"Duplicate chunk indices found in {chunk_dir}: {all_chunk_idxs}")
    chunk_path_by_idx = dict(zip(all_chunk_idxs, chunk_paths))

    if exclude_chunk_idxs is None:
        exclude_chunk_idxs = set()
    else:
        exclude_chunk_idxs = {int(idx) for idx in exclude_chunk_idxs}
    if validation_chunk_idxs is None:
        validation_chunk_idxs = set()
    else:
        validation_chunk_idxs = {int(idx) for idx in validation_chunk_idxs}
    invalid_excludes = sorted(idx for idx in exclude_chunk_idxs if idx not in chunk_path_by_idx)
    if invalid_excludes:
        raise ValueError(f"exclude_chunk_idxs contains indices not found in filenames: {invalid_excludes}")
    invalid_validation = sorted(idx for idx in validation_chunk_idxs if idx not in chunk_path_by_idx)
    if invalid_validation:
        raise ValueError(f"validation_chunk_idxs contains indices not found in filenames: {invalid_validation}")
    exclude_chunk_idxs = exclude_chunk_idxs | validation_chunk_idxs

    trainable_chunk_idxs = [idx for idx in all_chunk_idxs if idx not in exclude_chunk_idxs]
    total_chunks = len(trainable_chunk_idxs)
    if total_chunks == 0:
        raise ValueError("All chunk files are excluded. At least one chunk must remain for training.")
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
    if val_every_chunks is not None:
        val_every_chunks = int(val_every_chunks)
        if val_every_chunks < 1:
            raise ValueError("val_every_chunks must be >= 1")
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
        'recon_loss': [], 'KL_loss': [], 'total_loss': [], 'beta_eff': [],
        'val_global_chunk': [], 'val_recon_loss': [], 'val_KL_loss': [], 'val_total_loss': [],
        'val_beta_eff': [], 'val_kl_dim_mean': [], 'val_chunk_idxs': [], 'val_chunk_files': [],
        'time_chunk_load': [], 'time_load_wait': [], 'time_dataloader_init': [],
        'time_batch_fetch': [], 'time_h2d': [], 'time_forward': [], 'time_backward': [],
        'time_clip': [], 'time_step': [], 'time_loss_item': [], 'time_chunk_total': [], 'time_batches': [],
        'time_validation': []
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
        checkpoint_excluded_chunk_idxs = set(map(int, resume_checkpoint.get('excluded_chunk_idxs', [])))
        checkpoint_validation_chunk_idxs = set(map(int, resume_checkpoint.get('validation_chunk_idxs', [])))
        if checkpoint_excluded_chunk_idxs != exclude_chunk_idxs:
            raise ValueError(
                f"exclude_chunk_idxs가 checkpoint 설정({sorted(checkpoint_excluded_chunk_idxs)})과 다릅니다. "
                f"resume 입력값: {sorted(exclude_chunk_idxs)}"
            )
        if checkpoint_validation_chunk_idxs != validation_chunk_idxs:
            raise ValueError(
                f"validation_chunk_idxs가 checkpoint 설정({sorted(checkpoint_validation_chunk_idxs)})과 다릅니다. "
                f"resume 입력값: {sorted(validation_chunk_idxs)}"
            )
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
        for key, default_value in empty_chunk_history.items():
            chunk_loss_history.setdefault(key, default_value.copy())
        epoch_accum = resume_checkpoint.get('epoch_accum', epoch_accum)
        epoch_accum.setdefault('total_sum', 0.0)
        print(f"체크포인트 재개: {resume_path} | 완료 chunks={completed_chunks}")
    else:
        loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}
        chunk_loss_history = empty_chunk_history.copy()

    # start train
    train_start = time.perf_counter()
    target_chunks = completed_chunks + num_chunks
    print(f"GPU: {torch.cuda.get_device_name(device)} | cuda capability={torch.cuda.get_device_capability(device)}")
    print(
        f"학습 시작 | 이번 실행 chunks={num_chunks} | "
        f"진행 chunks={completed_chunks}->{target_chunks} | files/epoch={total_chunks} | "
        f"excluded={sorted(exclude_chunk_idxs)} | validation={sorted(validation_chunk_idxs)} | "
        f"val_every_chunks={val_every_chunks} | bn_chunks={bn_chunks} | warmup_chunks={warmup_chunks}"
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

            t0 = time.perf_counter()
            dataset, chunk_load_time = future.result()
            load_wait_time = time.perf_counter() - t0
            if offset + 1 < num_chunks:
                _, _, next_ci = chunk_info(global_chunk + 1)
                future = prefetcher.submit(load_chunk, next_ci)

            chunk_losses = train_one_chunk(dataset, global_chunk)
            chunk_losses[-1]['chunk_load'] = chunk_load_time
            chunk_losses[-1]['load_wait'] = load_wait_time
            record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses)

            del dataset

            current_chunks = global_chunk + 1
            should_validate = (
                len(validation_chunk_idxs) > 0
                and (
                    (val_every_chunks is not None and current_chunks % val_every_chunks == 0)
                    or offset == num_chunks - 1
                )
            )
            if should_validate:
                t0 = time.perf_counter()
                val_result = validate_chunks(current_chunks)
                _cuda_sync()
                validation_time = time.perf_counter() - t0
                record_validation_loss(val_result)
                chunk_loss_history['time_validation'][-1] = validation_time
                print(f"Validation time: {validation_time:.2f}s", flush=True)
            if current_chunks % total_chunks == 0:
                finish_epoch(epoch, current_epoch_start, current_chunks)

    total_time = time.perf_counter() - train_start

    n_logged = len(chunk_loss_history.get('chunk_idx', []))
    run_start_idx = max(0, n_logged - num_chunks)
    run_end_idx = n_logged
    if run_end_idx > run_start_idx:
        print("\n=== Time summary for this run ===")
        for key in [
            'chunk_load', 'load_wait', 'dataloader_init', 'batch_fetch', 'h2d',
            'forward', 'backward', 'clip', 'step', 'loss_item', 'validation', 'chunk_total'
        ]:
            values = chunk_loss_history.get(f'time_{key}', [])
            values = values[-num_chunks:]
            if len(values) == 0:
                continue
            values = np.asarray(values, dtype=np.float64)
            print(f"{key:16s} total={values.sum():9.2f}s | mean/chunk={values.mean():8.2f}s")
        batch_values = chunk_loss_history.get('time_batches', [])[-num_chunks:]
        if len(batch_values) > 0:
            print(f"{'batches':16s} total={int(np.sum(batch_values))} | mean/chunk={np.mean(batch_values):.1f}")
        print("=================================\n")

    t0 = time.perf_counter()
    save_checkpoint(target_chunks)
    final_save_time = time.perf_counter() - t0
    print(f"모델 저장 완료: {save_path} | save_time={final_save_time:.2f}s")
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