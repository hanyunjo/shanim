import time
import numpy as np
import torch
import h5py
import glob
import os
import re
import requests
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from e_2_CVAE import CVAE as BaseCVAE, freeze_batchnorm
from e_2_CVAE_barr_weight import CVAEBarrWeight



if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")

# _basic(eta에 B가 없는 버전), _B
CVAE = BaseCVAE
BS_CHUNK_DIR  = "/mnt/d/bs_chunks_correction/"
BS_ETA_PATH   = "/mnt/d/bs_eta_basic.h5"
BS_CLIP_CHUNK_DIR  = "/mnt/d/bs_clip_chunks_correction/"
BS_CLIP_ETA_PATH   = "/mnt/d/bs_eta_clip.h5"
HES_CHUNK_DIR = "/mnt/d/heston_chunks_correction/"
HES_ETA_PATH  = "/mnt/d/heston_eta_basic.h5"
HES_CLIP_CHUNK_DIR = "/mnt/d/heston_clip_chunks_correction/"
HES_CLIP_ETA_PATH  = "/mnt/d/heston_eta_clip.h5"


def alarm(message:str = "Jupyter 셀 실행 완료!"):
    webhook_url = "https://discord.com/api/webhooks/1522844432762671256/4h2AkQfGcD84AFu6FJSSFyV9FzQwsu0mFKTkTJ6ndUwD5MpfvdmXEDJJiA8tzM4Ba0P6"

    response = requests.post(
        webhook_url,
        json={
            "content": f"<@{427332366843772940}> {message}",
            "allowed_mentions": {
                "users": [427332366843772940]
            }
        },
        timeout=10
    )

    print(response.status_code)

# ──────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────
class GpuChunk:
    def __init__(self, x, eta):
        self.x = x
        self.eta = eta

    def __len__(self):
        return self.x.shape[0]


def load_chunk_file_to_gpu(chunk_path, etas, eta_min, eta_max):
    with h5py.File(chunk_path, 'r') as f:
        paths = f['paths'][:] # (N, 3) : [ori_idx, X_T, M_T]

    ori_idx = paths[:, 0].astype(np.int64, copy=False)
    x_np = paths[:, 1:3].astype(np.float32, copy=True)

    eta_np = etas[ori_idx].astype(np.float32, copy=True)
    eta_np = (eta_np - eta_min) / (eta_max - eta_min + 1e-8)

    del paths, ori_idx

    x_gpu = torch.from_numpy(x_np).to(device)
    eta_gpu = torch.from_numpy(eta_np).to(device)

    del x_np, eta_np
    return GpuChunk(x_gpu, eta_gpu)



class ChunkDataset(Dataset): # for Ram
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


def _chunk_sort_file_idx(chunk_path):
    filename = os.path.basename(chunk_path)
    match = re.search(r"chunk_(\d+)", filename)
    if match is None:
        raise ValueError(f"Cannot parse chunk index from filename: {filename}")
    return int(match.group(1))


def _normalize_cvae_type(cvae_type):
    aliases = {
        None: "base",
        "base": "base",
        "barr_weight": "barr_weight",
        "barrier_weight": "barr_weight",
        "normal_weight": "normal_weight",
    }
    if cvae_type not in aliases:
        raise ValueError("cvae_type must be 'base', 'barr_weight', or 'normal_weight'")
    return aliases[cvae_type]


def _make_cvae(cvae_type, dim_x, dim_eta, dim_z, hidden_dims, use_bn,
               weight_mode, weight_alpha, weight_h, weight_normalize, S0, K, B):
    cvae_type = _normalize_cvae_type(cvae_type)
    if cvae_type == "base":
        return BaseCVAE(
            dim_x=dim_x,
            dim_eta=dim_eta,
            dim_z=dim_z,
            hidden_dims=hidden_dims,
            use_bn=use_bn,
        )

    return CVAEBarrWeight(
        dim_x=dim_x,
        dim_eta=dim_eta,
        dim_z=dim_z,
        hidden_dims=hidden_dims,
        use_bn=use_bn,
        weight_mode=weight_mode,
        weight_alpha=weight_alpha,
        weight_h=weight_h,
        weight_normalize=weight_normalize,
        S0=S0,
        K=K,
        B=B,
    )


# ─────────────
# 2.train
# ─────────────
def train_chunk(model_type = 'hes', dim_z=8, hidden_dims=None, batch_size=1024,
                lr=1e-3, beta=1.0, warmup_chunks=None, use_bn=False, bn_chunks=None, 
                num_chunks=None, shuffle_chunks=True, save_path=None, resume_path=None,
                exclude_chunk_idxs=None, validation_chunk_idxs=None, val_every_chunks=10,
                memory_on_gpu=True, cvae_type="base", weight_mode="barrier_put",
                weight_alpha=3.0, weight_h=0.05, weight_normalize=True,
                S0=1.0, K=1.0, B=0.8):
    
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
        dataset = ChunkDataset(chunk_path_by_idx[int(ci)], etas, eta_min, eta_max)
        return dataset

    def load_chunk_gpu(ci):
        return load_chunk_file_to_gpu(chunk_path_by_idx[int(ci)], etas, eta_min, eta_max)

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
        cvae.train()
        bn_mode = apply_bn_mode(global_chunk)
        beta_eff = beta_eff_for_chunk(global_chunk)
        chunk_recon = 0.0
        chunk_kl = 0.0
        chunk_total = 0.0
        chunk_batches = 0

        if memory_on_gpu:
            n_rows = len(dataset)
            n_train = (n_rows // batch_size) * batch_size
            if n_train == 0:
                raise ValueError(f"batch_size={batch_size} is larger than chunk rows={n_rows}")

            for start in range(0, n_train, batch_size):
                end = start + batch_size
                x_batch = dataset.x[start:end]
                eta_batch = dataset.eta[start:end]

                recon_loss, kl_loss = cvae(x_batch, eta_batch)
                loss = recon_loss + beta_eff * kl_loss

                optimizer.zero_grad(set_to_none=True)
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

        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8, # batch 단위로 데이터를 미리 준비
                persistent_workers=True,
                prefetch_factor=1,
                pin_memory=True, # GPU가 빠르게 가져갈 수 있는 CPU 메모리 영역에 올려둠.
                drop_last=True
            )

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
                dataset = load_chunk_gpu(ci) if memory_on_gpu else load_chunk(ci)

                if memory_on_gpu:
                    n_rows = len(dataset)
                    for start in range(0, n_rows, batch_size):
                        end = min(start + batch_size, n_rows)
                        x_batch = dataset.x[start:end]
                        eta_batch = dataset.eta[start:end]

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

                    del dataset
                    torch.cuda.empty_cache()
                else:
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=8,
                        persistent_workers=True,
                        prefetch_factor=1,
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
            'cvae_type'   : cvae_type,
            'weight_config': weight_config,
            'loss_history': loss_history, # epoch별 평균 손실 기록
            'chunk_loss_history': chunk_loss_history,
            'num_chunks'  : num_chunks,
            'trained_chunks': current_chunks,
            'total_chunks': total_chunks,
            'shuffle_chunks': shuffle_chunks,
            'excluded_chunk_idxs': sorted(exclude_chunk_idxs),
            'validation_chunk_idxs': sorted(validation_chunk_idxs),
            'val_every_chunks': val_every_chunks,
            'memory_on_gpu': memory_on_gpu,
            'num_all_chunks': num_all_chunks,
            'all_chunk_idxs': all_chunk_idxs,
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
        chunk_loss_history['chunk_file'].append(os.path.basename(chunk_path_by_idx[int(ci)]))
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
            save_checkpoint(current_chunks)
            print(f"  중간 저장 완료 (epoch {epoch})")

    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    cvae_type = _normalize_cvae_type(cvae_type)
    weight_config = {
        'weight_mode': weight_mode,
        'weight_alpha': float(weight_alpha),
        'weight_h': float(weight_h),
        'weight_normalize': bool(weight_normalize),
        'S0': float(S0),
        'K': float(K),
        'B': float(B),
    }

    if model_type == 'hes':
        chunk_dir = HES_CHUNK_DIR
        eta_path  = HES_ETA_PATH
    elif model_type == 'bs':
        chunk_dir = BS_CHUNK_DIR
        eta_path  = BS_ETA_PATH
    elif model_type == 'bs_clip':
        chunk_dir = BS_CLIP_CHUNK_DIR
        eta_path  = BS_CLIP_ETA_PATH
    elif model_type == 'hes_clip':
        chunk_dir = HES_CLIP_CHUNK_DIR
        eta_path  = HES_CLIP_ETA_PATH
    else:
        raise ValueError("model_type must be 'hes' or 'bs'")

    dim_x = 2   # (X_T, M_T)

    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.h5")), key=_chunk_sort_file_idx)
    num_all_chunks = len(chunk_paths)
    if num_all_chunks == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    all_chunk_idxs = [_chunk_sort_file_idx(path) for path in chunk_paths]
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

        checkpoint_cvae_type = _normalize_cvae_type(resume_checkpoint.get('cvae_type', 'base'))
        if checkpoint_cvae_type != cvae_type:
            raise ValueError(
                f"checkpoint cvae_type={checkpoint_cvae_type}, current cvae_type={cvae_type}. "
                "다른 CVAE loss로 이어서 학습하려면 새 save_path로 별도 실험을 시작하세요."
            )
        if cvae_type in ('barr_weight', 'normal_weight'):
            checkpoint_weight_config = resume_checkpoint.get('weight_config', {})
            for key, value in weight_config.items():
                if key in checkpoint_weight_config and checkpoint_weight_config[key] != value:
                    raise ValueError(
                        f"checkpoint weight_config[{key}]={checkpoint_weight_config[key]}, "
                        f"current {key}={value}"
                    )

    # 모델 생성
    cvae = _make_cvae(
        cvae_type,
        dim_x=dim_x,
        dim_eta=dim_eta,
        dim_z=dim_z,
        hidden_dims=hidden_dims,
        use_bn=use_bn,
        weight_mode=weight_mode,
        weight_alpha=weight_alpha,
        weight_h=weight_h,
        weight_normalize=weight_normalize,
        S0=S0,
        K=K,
        B=B,
    )
    cvae.bn_chunks = bn_chunks
    cvae.to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    completed_chunks = 0
    epoch_accum = {'recon_sum': 0.0, 'kl_sum': 0.0, 'total_sum': 0.0, 'n_batches': 0}
    empty_chunk_history = {
        'epoch': [], 'global_chunk': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [],
        'recon_loss': [], 'KL_loss': [], 'total_loss': [], 'beta_eff': [],
        'val_global_chunk': [], 'val_recon_loss': [], 'val_KL_loss': [], 'val_total_loss': [],
        'val_beta_eff': [], 'val_kl_dim_mean': [], 'val_chunk_idxs': [], 'val_chunk_files': []
    }

    if resume_path is not None: # load resume model
        cvae.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
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

    print(f"learning rate : {lr}")
    # start train
    train_start = time.perf_counter()
    target_chunks = completed_chunks + num_chunks
    print(
        f"학습 시작 | 이번 실행 chunks={num_chunks} | "
        f"진행 chunks={completed_chunks}->{target_chunks} | files/epoch={total_chunks} | "
        f"excluded={sorted(exclude_chunk_idxs)} | validation={sorted(validation_chunk_idxs)} | "
        f"val_every_chunks={val_every_chunks} | bn_chunks={bn_chunks} | warmup_chunks={warmup_chunks} | "
        f"memory_on_gpu={memory_on_gpu} | cvae_type={cvae_type}"
    )
    if cvae_type in ('barr_weight', 'normal_weight'):
        print(f"weighted recon config: {weight_config}")

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

            if memory_on_gpu:
                cpu_dataset = dataset
                dataset = GpuChunk(cpu_dataset.x.to(device), cpu_dataset.eta.to(device))
                del cpu_dataset

            chunk_losses = train_one_chunk(dataset, global_chunk)
            record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses)

            del dataset
            if memory_on_gpu:
                torch.cuda.empty_cache()

            current_chunks = global_chunk + 1
            should_validate = (
                len(validation_chunk_idxs) > 0
                and (
                    (val_every_chunks is not None and current_chunks % val_every_chunks == 0)
                    or offset == num_chunks - 1
                )
            )
            if should_validate:
                record_validation_loss(validate_chunks(current_chunks))
            if current_chunks % total_chunks == 0:
                finish_epoch(epoch, current_epoch_start, current_chunks)

    total_time = time.perf_counter() - train_start
    save_checkpoint(target_chunks)
    print(f"모델 저장 완료: {save_path}")
    print(f"총 학습 시간: {total_time/60:.2f}분 ({total_time/3600:.2f}시간)")

    return cvae, loss_history, eta_min, eta_max