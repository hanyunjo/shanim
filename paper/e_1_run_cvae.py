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
from e_2_CVAE_UT import CVAE as UTCVAE
from e_2_CVAE_UT import CVAEBarrWeight as UTCVAEBarrWeight



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
HES_CLIP_CHUNK_DIR = "/mnt/d/heston_clip_x2_chunks_correction/"
HES_CLIP_ETA_PATH  = "/mnt/d/heston_eta_clip_x2.h5"


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


def load_chunk_file_to_gpu(chunk_path, etas, eta_min, eta_max, include_m=True):
    with h5py.File(chunk_path, 'r') as f:
        paths = f['paths'][:] # (N, 3) : [ori_idx, X_T, M_T]

    ori_idx = paths[:, 0].astype(np.int64, copy=False)
    x_stop = 3 if include_m else 2
    x_np = paths[:, 1:x_stop].astype(np.float32, copy=True)

    eta_np = etas[ori_idx].astype(np.float32, copy=True)
    eta_np = (eta_np - eta_min) / (eta_max - eta_min + 1e-8)

    del paths, ori_idx

    x_gpu = torch.from_numpy(x_np).to(device)
    eta_gpu = torch.from_numpy(eta_np).to(device)

    del x_np, eta_np
    return GpuChunk(x_gpu, eta_gpu)



class ChunkDataset(Dataset): # for Ram
    def __init__(self, chunk_path, etas, eta_min, eta_max, include_m=True):

        with h5py.File(chunk_path, 'r') as f:
            paths = f['paths'][:] # (N, 3) : [ori_idx, X_T, M_T]

        ori_idx = paths[:, 0].astype(int)
        x_stop = 3 if include_m else 2
        x_np = paths[:, 1:x_stop].astype(np.float32, copy=True)

        eta_matched = etas[ori_idx].astype(np.float32)
        eta_matched = (eta_matched - eta_min) / (eta_max - eta_min + 1e-8)

        self.x = torch.from_numpy(x_np)  # include_m=False: (N, 1), True: (N, 2)
        self.eta = torch.from_numpy(eta_matched)   # (N, dim_eta)

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
        "add_put_loss": "add_put_loss",
        "additive_put_loss": "add_put_loss",
    }
    if cvae_type not in aliases:
        raise ValueError("cvae_type must be 'base', 'barr_weight', 'normal_weight', or 'add_put_loss'")
    return aliases[cvae_type]


def _normalize_target_parameterization(target_parameterization):
    aliases = {
        None: "mt",
        "mt": "mt",
        "direct": "mt",
        "base": "mt",
        "legacy": "mt",
        "ut": "ut",
        "transformed": "ut",
        "support_aware": "ut",
    }
    normalized = str(target_parameterization).lower() if target_parameterization is not None else None
    if normalized not in aliases:
        raise ValueError(
            "target_parameterization must be 'mt' or 'ut'."
        )
    return aliases[normalized]


def _checkpoint_target_parameterization(checkpoint):
    saved_value = checkpoint.get("target_parameterization")
    if saved_value is not None:
        return _normalize_target_parameterization(saved_value)

    model_coordinates = checkpoint.get("model_coordinates")
    if model_coordinates is None:
        model_state = checkpoint.get("model_state", {})
        extra_state = model_state.get("_extra_state", {})
        model_coordinates = extra_state.get("model_coordinates")

    if model_coordinates is None or list(model_coordinates) in (["X_T"], ["X_T", "M_T"]):
        return "mt"
    if list(model_coordinates) == ["X_T", "U_T"]:
        return "ut"
    raise ValueError(
        f"Unsupported checkpoint model_coordinates: {model_coordinates!r}"
    )


def _make_cvae(target_parameterization, cvae_type, dim_x, dim_eta, dim_z,
               hidden_dims, use_bn, residual_blocks,
               weight_mode, weight_alpha, weight_mode2, weight_alpha2,
               weight_h, weight_normalize, S0, K, B):
    target_parameterization = _normalize_target_parameterization(
        target_parameterization
    )
    cvae_type = _normalize_cvae_type(cvae_type)
    if target_parameterization == "ut":
        base_class = UTCVAE
        weighted_class = UTCVAEBarrWeight
    else:
        base_class = BaseCVAE
        weighted_class = CVAEBarrWeight

    if cvae_type == "base":
        return base_class(
            dim_x=dim_x,
            dim_eta=dim_eta,
            dim_z=dim_z,
            hidden_dims=hidden_dims,
            use_bn=use_bn,
            residual_blocks=residual_blocks,
        )

    return weighted_class(
        dim_x=dim_x,
        dim_eta=dim_eta,
        dim_z=dim_z,
        hidden_dims=hidden_dims,
        use_bn=use_bn,
        residual_blocks=residual_blocks,
        weight_mode=weight_mode,
        weight_alpha=weight_alpha,
        weight_mode2=weight_mode2,
        weight_alpha2=weight_alpha2,
        weight_h=weight_h,
        weight_normalize=weight_normalize,
        cvae_type=cvae_type,
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
                init_path=None, exclude_chunk_idxs=None, validation_chunk_idxs=None, 
                val_every_chunks=10, memory_on_gpu=True, cvae_type="base",
                target_parameterization="mt",
                weight_mode="barrier_put", weight_alpha=3.0, weight_mode2=None,
                weight_alpha2=0.0, weight_h=0.05, weight_normalize=True,
                S0=1.0, K=1.0, B=0.8, tmp_save_every_chunks=10,
                include_m=True, residual_blocks=0):
    
    def chunk_order_for_epoch(epoch):
        chunk_indices = np.array(trainable_chunk_idxs, dtype=int)
        if shuffle_chunks:
            rng = np.random.default_rng(epoch) # epoch = seed
            return rng.permutation(chunk_indices)
        return chunk_indices

    def chunk_info(global_chunk):
        epoch = global_chunk // total_chunks + 1
        chunk_pos = global_chunk % total_chunks
        chunk_order = chunk_order_for_epoch(epoch)
        ci = int(chunk_order[chunk_pos])
        return epoch, chunk_pos, ci

    def load_chunk(ci):
        dataset = ChunkDataset(
            chunk_path_by_idx[int(ci)], etas, eta_min, eta_max,
            include_m=include_m,
        )
        return dataset

    def load_chunk_gpu(ci):
        return load_chunk_file_to_gpu(
            chunk_path_by_idx[int(ci)], etas, eta_min, eta_max,
            include_m=include_m,
        )

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

    def _checkpoint_path_with_suffix(path, suffix):
        root, ext = os.path.splitext(path)
        if ext:
            return f"{root}{suffix}{ext}"
        return f"{path}{suffix}"

    def _checkpoint_payload(current_chunks):
        completed_epochs = current_chunks // total_chunks
        payload = {
            'model_state' : cvae.state_dict(),
            'eta_min'     : eta_min,
            'eta_max'     : eta_max,
            'dim_x'       : dim_x,
            'include_m'   : include_m,
            'dim_eta'     : dim_eta,
            'dim_z'       : dim_z,
            'hidden_dims' : hidden_dims,
            'residual_blocks': residual_blocks,
            'residual_layout': cvae.residual_layout,
            'use_bn'      : use_bn,
            'bn_chunks'   : bn_chunks,
            'warmup_chunks': warmup_chunks,
            'cvae_type'   : cvae_type,
            'target_parameterization': target_parameterization,
            'weight_config': weight_config,
            'weight_config_history': weight_config_history,
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
            'tmp_save_every_chunks': tmp_save_every_chunks,
        }
        if hasattr(cvae, "checkpoint_metadata"):
            payload.update(cvae.checkpoint_metadata())
        else:
            payload.update({
                "model_coordinates": ["X_T", "M_T"] if include_m else ["X_T"],
                "physical_coordinates": ["X_T", "M_T"] if include_m else ["X_T"],
                "support_parameterization": "model output M_T" if include_m else None,
                "path_statistic": "running_minimum" if include_m else None,
            })
        return payload

    def save_checkpoint(current_chunks, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = save_path
        writing_path = f"{checkpoint_path}.writing"
        try:
            torch.save(_checkpoint_payload(current_chunks), writing_path)
            os.replace(writing_path, checkpoint_path)
        except Exception:
            if os.path.exists(writing_path):
                try:
                    os.remove(writing_path)
                except OSError:
                    pass
            raise

    def save_tmp_checkpoint(current_chunks, tmp_checkpoint_path):
        save_checkpoint(current_chunks, tmp_checkpoint_path)
        print(f"tmp checkpoint 저장 완료: {tmp_checkpoint_path} | 완료 chunks={current_chunks}", flush=True)

    def remove_tmp_checkpoint(tmp_checkpoint_path):
        if not os.path.exists(tmp_checkpoint_path):
            return
        try:
            os.remove(tmp_checkpoint_path)
            print(f"tmp checkpoint 삭제 완료: {tmp_checkpoint_path}", flush=True)
        except OSError as exc:
            print(f"tmp checkpoint 삭제 실패: {exc!r}", flush=True)

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

    if not isinstance(residual_blocks, (int, np.integer)) or isinstance(residual_blocks, bool):
        raise TypeError("residual_blocks must be a non-negative integer.")
    residual_blocks = int(residual_blocks)
    if residual_blocks < 0:
        raise ValueError("residual_blocks must be >= 0.")

    if not isinstance(include_m, (bool, np.bool_)):
        raise TypeError("include_m must be True or False.")
    include_m = bool(include_m)

    target_parameterization = _normalize_target_parameterization(
        target_parameterization
    )
    if target_parameterization == "ut" and not include_m:
        raise ValueError(
            "target_parameterization='ut' requires include_m=True because the raw "
            "training batch must contain [X_T, M_T]."
        )

    cvae_type = _normalize_cvae_type(cvae_type)
    barrier_weight_modes = {"barrier_put", "barrierput", "barrier_near", "barriernear"}
    weight_mode_normalized = str(weight_mode).lower()
    weight_mode2_normalized = None if weight_mode2 is None else str(weight_mode2).lower()
    weight_uses_m = (
        cvae_type == "barr_weight"
        or (
            cvae_type in ("normal_weight", "add_put_loss")
            and weight_mode_normalized in barrier_weight_modes
        )
        or (
            cvae_type == "normal_weight"
            and weight_mode2_normalized in barrier_weight_modes
        )
    )
    if not include_m and weight_uses_m:
        raise ValueError(
            "include_m=False cannot be used with an M_T-dependent reconstruction loss. "
            "Use cvae_type='base', or choose an X_T-only mode such as weight_mode='all_put'."
        )
    weight_config = {
        'weight_mode': weight_mode,
        'weight_alpha': float(weight_alpha),
        'weight_mode2': weight_mode2,
        'weight_alpha2': float(weight_alpha2),
        'weight_h': float(weight_h),
        'weight_normalize': bool(weight_normalize),
        'S0': float(S0),
        'K': float(K),
        'B': float(B),
    }
    weight_config_history = [{'trained_chunks': 0, 'weight_config': weight_config.copy()}]
    weight_alpha_changed_on_resume = False

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

    dim_x = 2 if include_m else 1  # True: (X_T, M_T), False: (X_T,)

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
    if tmp_save_every_chunks is not None:
        tmp_save_every_chunks = int(tmp_save_every_chunks)
        if tmp_save_every_chunks < 1:
            raise ValueError("tmp_save_every_chunks must be >= 1 or None")
    if resume_path is not None and init_path is not None:
        raise ValueError("resume_path and init_path cannot be used together.")
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

        checkpoint_target_parameterization = _checkpoint_target_parameterization(
            resume_checkpoint
        )
        if checkpoint_target_parameterization != target_parameterization:
            raise ValueError(
                "checkpoint target_parameterization="
                f"{checkpoint_target_parameterization}, current="
                f"{target_parameterization}. An MT checkpoint cannot be resumed "
                "as UT, and a UT checkpoint cannot be resumed as MT."
            )

        checkpoint_use_bn = bool(resume_checkpoint.get('use_bn', False))
        if checkpoint_use_bn != bool(use_bn):
            print(f"use_bn을 checkpoint 설정({checkpoint_use_bn})으로 맞춥니다.")
        use_bn = checkpoint_use_bn

        checkpoint_hidden_dims = resume_checkpoint.get('hidden_dims', hidden_dims)
        if checkpoint_hidden_dims != hidden_dims:
            print(f"hidden_dims를 checkpoint 설정({checkpoint_hidden_dims})으로 맞춥니다.")
        hidden_dims = checkpoint_hidden_dims

        checkpoint_residual_blocks = int(resume_checkpoint.get('residual_blocks', 0))
        if checkpoint_residual_blocks != residual_blocks:
            print(
                f"residual_blocks를 checkpoint 설정({checkpoint_residual_blocks})으로 맞춥니다."
            )
        residual_blocks = checkpoint_residual_blocks
        checkpoint_residual_layout = resume_checkpoint.get('residual_layout')
        if (
            residual_blocks > 0
            and checkpoint_residual_layout != BaseCVAE.RESIDUAL_LAYOUT
        ):
            raise ValueError(
                "This checkpoint uses a different residual layout. "
                "It cannot be resumed with the paired equal-width residual layout; "
                "start a new experiment instead."
            )

        checkpoint_dim_z = int(resume_checkpoint.get('dim_z', dim_z))
        if checkpoint_dim_z != dim_z:
            print(f"dim_z를 checkpoint 설정({checkpoint_dim_z})으로 맞춥니다.")
        dim_z = checkpoint_dim_z

        checkpoint_dim_x = int(resume_checkpoint.get('dim_x', dim_x))
        if checkpoint_dim_x != dim_x:
            raise ValueError(f"checkpoint dim_x={checkpoint_dim_x}, current dim_x={dim_x}")

        checkpoint_include_m = bool(
            resume_checkpoint.get('include_m', checkpoint_dim_x == 2)
        )
        if checkpoint_include_m != include_m:
            raise ValueError(
                f"checkpoint include_m={checkpoint_include_m}, current include_m={include_m}"
            )

        checkpoint_dim_eta = int(resume_checkpoint.get('dim_eta', dim_eta))
        if checkpoint_dim_eta != dim_eta:
            raise ValueError(f"checkpoint dim_eta={checkpoint_dim_eta}, current eta dim={dim_eta}")

        checkpoint_cvae_type = _normalize_cvae_type(resume_checkpoint.get('cvae_type', 'base'))
        if checkpoint_cvae_type != cvae_type:
            raise ValueError(
                f"checkpoint cvae_type={checkpoint_cvae_type}, current cvae_type={cvae_type}. "
                "다른 CVAE loss로 이어서 학습하려면 새 save_path로 별도 실험을 시작하세요."
            )
        if cvae_type in ('barr_weight', 'normal_weight', 'add_put_loss'):
            checkpoint_weight_config = resume_checkpoint.get('weight_config', {})
            for key, value in weight_config.items():
                if key in ('weight_alpha', 'weight_alpha2'):
                    continue
                if key in checkpoint_weight_config and checkpoint_weight_config[key] != value:
                    raise ValueError(
                        f"checkpoint weight_config[{key}]={checkpoint_weight_config[key]}, "
                        f"current {key}={value}"
                    )
            if (
                (
                    'weight_alpha' in checkpoint_weight_config
                    and checkpoint_weight_config['weight_alpha'] != weight_config['weight_alpha']
                )
                or (
                    'weight_alpha2' in checkpoint_weight_config
                    and checkpoint_weight_config['weight_alpha2'] != weight_config['weight_alpha2']
                )
            ):
                weight_alpha_changed_on_resume = True
                print(
                    f"weight_alpha 변경 resume: "
                    f"{checkpoint_weight_config.get('weight_alpha')} -> {weight_config['weight_alpha']}, "
                    f"weight_alpha2: "
                    f"{checkpoint_weight_config.get('weight_alpha2')} -> {weight_config['weight_alpha2']}"
                )

    # 모델 생성
    cvae = _make_cvae(
        target_parameterization,
        cvae_type,
        dim_x=dim_x,
        dim_eta=dim_eta,
        dim_z=dim_z,
        hidden_dims=hidden_dims,
        use_bn=use_bn,
        residual_blocks=residual_blocks,
        weight_mode=weight_mode,
        weight_alpha=weight_alpha,
        weight_mode2=weight_mode2,
        weight_alpha2=weight_alpha2,
        weight_h=weight_h,
        weight_normalize=weight_normalize,
        S0=S0,
        K=K,
        B=B,
    )
    cvae.bn_chunks = bn_chunks
    cvae.include_m = include_m
    cvae.to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)

    if init_path is not None:
        init_checkpoint = torch.load(init_path, map_location=device, weights_only=False)
        init_target_parameterization = _checkpoint_target_parameterization(
            init_checkpoint
        )
        if init_target_parameterization != target_parameterization:
            raise ValueError(
                "init checkpoint target_parameterization="
                f"{init_target_parameterization}, current={target_parameterization}. "
                "Weights trained in MT and UT coordinates are not interchangeable."
            )
        init_residual_blocks = int(init_checkpoint.get('residual_blocks', 0))
        if init_residual_blocks != residual_blocks:
            raise ValueError(
                f"init checkpoint residual_blocks={init_residual_blocks}, "
                f"current residual_blocks={residual_blocks}. "
                "init_path requires the same residual architecture."
            )
        init_residual_layout = init_checkpoint.get('residual_layout')
        if (
            residual_blocks > 0
            and init_residual_layout != cvae.residual_layout
        ):
            raise ValueError(
                "The init checkpoint uses a different residual layout. "
                "It is incompatible with the paired equal-width residual layout."
            )
        init_state_dict = init_checkpoint["model_state"]
        if any(key.startswith("module.") for key in init_state_dict):
            init_state_dict = {
                key.replace("module.", "", 1): value
                for key, value in init_state_dict.items()
            }
        cvae.load_state_dict(init_state_dict, strict=True)
        print(
            f"초기 가중치 로드: {init_path} | "
            f"source cvae_type={init_checkpoint.get('cvae_type', 'base')} -> "
            f"target cvae_type={cvae_type}"
        )

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
        weight_config_history = list(resume_checkpoint.get('weight_config_history', []))
        if not weight_config_history and 'weight_config' in resume_checkpoint:
            weight_config_history = [{
                'trained_chunks': 0,
                'weight_config': dict(resume_checkpoint['weight_config']),
            }]
        if weight_alpha_changed_on_resume:
            weight_config_history.append({
                'trained_chunks': completed_chunks,
                'weight_config': weight_config.copy(),
                'note': 'weight_alpha changed on resume',
            })
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
    tmp_save_path = _checkpoint_path_with_suffix(save_path, "_tmp")
    print(
        f"학습 시작 | 이번 실행 chunks={num_chunks} | "
        f"진행 chunks={completed_chunks}->{target_chunks} | files/epoch={total_chunks} | "
        f"excluded={sorted(exclude_chunk_idxs)} | validation={sorted(validation_chunk_idxs)} | "
        f"val_every_chunks={val_every_chunks} | bn_chunks={bn_chunks} | warmup_chunks={warmup_chunks} | "
        f"memory_on_gpu={memory_on_gpu} | cvae_type={cvae_type} | "
        f"target_parameterization={target_parameterization} | "
        f"tmp_save_every_chunks={tmp_save_every_chunks}"
    )
    if cvae_type in ('barr_weight', 'normal_weight', 'add_put_loss'):
        print(f"weighted recon config: {weight_config}")

    current_chunks = completed_chunks
    current_epoch_start = time.perf_counter()
    try:
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
                    print(f"Validation 시작 @ chunk {current_chunks}", flush=True)
                    record_validation_loss(validate_chunks(current_chunks))
                if current_chunks % total_chunks == 0:
                    finish_epoch(epoch, current_epoch_start, current_chunks)
                if (
                    tmp_save_every_chunks is not None
                    and (current_chunks - completed_chunks) % tmp_save_every_chunks == 0
                ):
                    save_tmp_checkpoint(current_chunks, tmp_save_path)
                if offset + 1 < num_chunks:
                    _, _, next_ci = chunk_info(global_chunk + 1)
                    future = prefetcher.submit(load_chunk, next_ci)
    except (KeyboardInterrupt, Exception) as exc:
        total_time = time.perf_counter() - train_start
        print(
            f"\n학습 중단/에러 감지 ({type(exc).__name__}) | "
            f"저장 기준 완료 chunks={current_chunks}/{target_chunks}",
            flush=True,
        )
        if current_chunks > completed_chunks:
            interrupt_save_path = _checkpoint_path_with_suffix(save_path, f"_interrupt{current_chunks}")
            try:
                save_checkpoint(current_chunks, interrupt_save_path)
                print(f"interrupt checkpoint 저장 완료: {interrupt_save_path}", flush=True)
            except Exception as save_exc:
                print(f"interrupt checkpoint 저장 실패: {save_exc!r}", flush=True)
        else:
            print("이번 실행에서 완료된 새 chunk가 없어 새 checkpoint를 저장하지 않았습니다.", flush=True)
        print(
            f"중단 전 학습 시간: {total_time/60:.2f}분 ({total_time/3600:.2f}시간)",
            flush=True,
        )
        raise

    total_time = time.perf_counter() - train_start
    save_checkpoint(target_chunks)
    remove_tmp_checkpoint(tmp_save_path)
    print(f"모델 저장 완료: {save_path}")
    print(f"총 학습 시간: {total_time/60:.2f}분 ({total_time/3600:.2f}시간)")

    return cvae, loss_history, eta_min, eta_max