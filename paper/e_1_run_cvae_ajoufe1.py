import time
import numpy as np
import torch
import h5py
import glob
import os
import re
import traceback
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from e_2_CVAE import *

if not torch.cuda.is_available():
    raise ValueError("Cannot use GPU cuda")
device = torch.device("cuda")

# _basic(eta에 B가 없는 버전), _B
BS_CHUNK_DIR  = os.path.expanduser("~/yunjo/bs_chunks_correction/")
BS_ETA_PATH   = os.path.expanduser("~/yunjo/bs_eta_basic.h5")
HES_CHUNK_DIR = os.path.expanduser("~/yunjo/chunks/")
HES_ETA_PATH  = os.path.expanduser("~/yunjo/heston_eta_basic.h5")


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
# Distributed data parallel(multi GPU train)
# ─────────────
def _ddp_setup():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl") # gpu process 간 통신을 위한 backend로 nccl 사용

    # node_rank = server 번호
    rank = dist.get_rank() # rank 번호 = 전체 gpu 번호, 
    local_rank = int(os.environ["LOCAL_RANK"]) # server별 gpu 번호
    world_size = dist.get_world_size() # 전체 프로세스 개수 = rank
    torch.cuda.set_device(local_rank)
    local_device = torch.device(f"cuda:{local_rank}")

    return rank, local_rank, world_size, local_device


def _rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def _ddp_log(rank, msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [rank {rank}] {msg}", flush=True)

def _load_plain_state_dict(model, state_dict):
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


def _load_chunk_dataset(chunk_path, etas, eta_min, eta_max, rank, epoch, chunk_pos, n_chunks, ci):
    _ddp_log(rank, f"epoch={epoch} chunk_pos={chunk_pos}/{n_chunks-1} ci={ci} load start")
    dataset = ChunkDataset(chunk_path, etas, eta_min, eta_max)
    _ddp_log(rank, f"epoch={epoch} chunk_pos={chunk_pos} dataset loaded len={len(dataset)}")
    return dataset


def train_chunk_ddp(model_type='hes', dim_z=8, hidden_dims=None, batch_size=1024,
                    lr=1e-3, beta=1.0, warmup_chunks=None, save_path=None,
                    use_bn=False, bn_chunks=None, num_chunks=None, shuffle_chunks=True, num_workers=4, prefetch_factor=2, seed=1234, overwrite=False,
                    resume_path=None):
    # batch size is batch_size * number_of_gpus.

    rank, local_rank, world_size, local_device = _ddp_setup()

    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    if model_type == 'hes':
        chunk_dir = HES_CHUNK_DIR
        eta_path  = HES_ETA_PATH
    else:
        chunk_dir = BS_CHUNK_DIR
        eta_path  = BS_ETA_PATH

    dim_x = 2   # always train (X_T, M_T); vanilla pricing ignores M_T

    chunk_paths = sorted(
        glob.glob(os.path.join(os.path.expanduser(chunk_dir), "*.h5")),
        key=_chunk_sort_key,
    )
    total_chunks = len(chunk_paths)
    if total_chunks == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    if num_chunks is None:
        num_chunks = total_chunks
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
        save_path = f"result/cvae/{model_type}/cvae_{model_type}_{dim_z}_{hidden_dims[0]}_{batch_size}_{bn_chunks}_{lr}_{beta}_{warmup_chunks}_chunk{num_chunks}.pt"
    save_path = os.path.expanduser(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    resume_path = os.path.expanduser(resume_path) if resume_path is not None else None
    if os.path.exists(save_path) and not overwrite and save_path != resume_path:
        raise FileExistsError(
            f"{save_path} already exists. Use overwrite=True or choose a different save_path."
        )

    etas, eta_min, eta_max = compute_eta_stats(os.path.expanduser(eta_path))
    dim_eta = etas.shape[1]

    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=local_device, weights_only=False)

    checkpoint = resume_checkpoint
    if checkpoint is not None:
        checkpoint_use_bn = bool(checkpoint.get('use_bn', False))
        if checkpoint_use_bn != bool(use_bn):
            _rank0_print(rank, f"use_bn을 checkpoint 설정({checkpoint_use_bn})으로 맞춥니다.")
        use_bn = checkpoint_use_bn
        checkpoint_bn_chunks = checkpoint.get('bn_chunks', None)
        checkpoint_warmup_chunks = checkpoint.get('warmup_chunks', warmup_chunks)
    else:
        use_bn = bool(use_bn)

    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims, use_bn=use_bn).to(local_device)
    cvae.bn_chunks = bn_chunks

    completed_chunks = 0
    epoch_accum = {'recon_sum': 0.0, 'kl_sum': 0.0, 'total_sum': 0.0, 'n_batches': 0.0}
    empty_chunk_history = {
        'epoch': [], 'global_chunk': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [],
        'recon_loss': [], 'KL_loss': [], 'total_loss': [], 'beta_eff': []
    }

    if resume_checkpoint is not None:
        _load_plain_state_dict(cvae, resume_checkpoint['model_state'])
        completed_chunks = int(resume_checkpoint.get('trained_chunks', len(resume_checkpoint.get('chunk_loss_history', {}).get('chunk_idx', []))))
        if checkpoint_bn_chunks is None:
            if use_bn and bn_chunks is not None and completed_chunks > bn_chunks:
                raise ValueError(
                    f"이전 checkpoint에는 bn_chunks가 없고 이미 {completed_chunks} chunks가 BN으로 학습됐습니다. "
                    f"bn_chunks={bn_chunks}로는 과거 BN 적용을 되돌릴 수 없어 resume할 수 없습니다."
                )
            if bn_chunks is not None:
                _rank0_print(rank, f"기존 checkpoint에 bn_chunks가 없어 새 설정({bn_chunks})으로 이어서 저장합니다.")
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
        _rank0_print(rank, f"체크포인트 재개: {resume_path} | 완료 chunks={completed_chunks}")

    cvae = DDP(cvae, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if resume_checkpoint is not None:
        if 'optimizer_state' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state'])
        if 'scheduler_state' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state'])
        loss_history = resume_checkpoint.get('loss_history', {'recon_loss': [], 'KL_loss': [], 'total_loss': []})
        chunk_loss_history = resume_checkpoint.get('chunk_loss_history', empty_chunk_history.copy())
    else:
        loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}
        chunk_loss_history = empty_chunk_history.copy()

    if rank == 0:
        chunk_loss_history.setdefault('global_chunk', [])
        if len(chunk_loss_history['global_chunk']) < len(chunk_loss_history.get('chunk_idx', [])):
            chunk_loss_history['global_chunk'] = list(range(len(chunk_loss_history.get('chunk_idx', []))))

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

    def trained_chunk_indices_for_epoch(epoch):
        epochs = chunk_loss_history.get('epoch', [])
        chunk_indices = chunk_loss_history.get('chunk_idx', [])
        return {
            int(ci)
            for ep, ci in zip(epochs, chunk_indices)
            if int(ep) == int(epoch)
        }

    def select_chunk_plan(start_chunk, n_chunks):
        plan = []
        selected_by_epoch = {}
        current_chunk = start_chunk

        while len(plan) < n_chunks:
            epoch = current_chunk // total_chunks + 1
            chunk_pos = current_chunk % total_chunks
            used = trained_chunk_indices_for_epoch(epoch) | selected_by_epoch.get(epoch, set())
            remaining_order = [
                int(ci)
                for ci in chunk_order_for_epoch(epoch)
                if int(ci) not in used
            ]

            if not remaining_order:
                current_chunk = epoch * total_chunks
                continue

            ci = remaining_order[0]
            plan.append((current_chunk, epoch, chunk_pos, ci))
            selected_by_epoch.setdefault(epoch, set()).add(ci)
            current_chunk += 1

        return plan

    planned_chunks = select_chunk_plan(completed_chunks, num_chunks)
    run_start_chunk = completed_chunks
    target_chunks = planned_chunks[-1][0] + 1
    _rank0_print(
        rank,
        f"DDP 학습 시작 | world_size={world_size} | per_gpu_batch={batch_size} | "
        f"global_batch={batch_size * world_size} | 이번 실행 chunks={len(planned_chunks)} | "
        f"진행 chunks={run_start_chunk}->{target_chunks} | files/epoch={total_chunks} | "
        f"shuffle_chunks={shuffle_chunks} | bn_chunks={bn_chunks} | warmup_chunks={warmup_chunks}"
    )

    train_start = time.perf_counter()

    def save_checkpoint(current_chunks):
        if rank != 0:
            return
        completed_epochs = current_chunks // total_chunks
        torch.save({
            'epoch'          : completed_epochs,
            'trained_chunks' : current_chunks,
            'model_state'    : cvae.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'eta_min'        : eta_min,
            'eta_max'        : eta_max,
            'dim_x'          : dim_x,
            'dim_eta'        : dim_eta,
            'dim_z'          : dim_z,
            'hidden_dims'    : hidden_dims,
            'use_bn'         : use_bn,
            'bn_chunks'      : bn_chunks,
            'warmup_chunks'  : warmup_chunks,
            'loss_history'   : loss_history,
            'chunk_loss_history': chunk_loss_history,
            'num_chunks'     : num_chunks,
            'total_chunks'   : total_chunks,
            'shuffle_chunks' : shuffle_chunks,
            'epoch_accum'    : epoch_accum,
            'batch_size'     : batch_size,
            'lr'             : lr,
            'beta'           : beta,
        }, save_path)

    def record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses, bn_mode):
        if rank != 0:
            return
        chunk_avg_recon, chunk_avg_kl, chunk_avg_total, beta_eff = chunk_losses
        chunk_loss_history.setdefault('beta_eff', [])
        chunk_loss_history['epoch'].append(epoch)
        chunk_loss_history['global_chunk'].append(global_chunk)
        chunk_loss_history['chunk_pos'].append(chunk_pos)
        chunk_loss_history['chunk_idx'].append(ci)
        chunk_loss_history['chunk_file'].append(os.path.basename(chunk_paths[ci]))
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

    def apply_bn_mode(global_chunk):
        if not use_bn:
            return "off"
        if bn_chunks is not None and global_chunk >= bn_chunks:
            freeze_batchnorm(cvae.module)
            return "frozen"
        cvae.train()
        return "train"

    def beta_eff_for_chunk(global_chunk):
        if warmup_chunks is None or warmup_chunks == 0:
            return float(beta)
        ratio = min(1.0, float(global_chunk + 1) / float(warmup_chunks))
        return float(beta) * ratio

    def finish_epoch(epoch, epoch_start, current_chunks):
        scheduler.step()
        if rank == 0:
            n_batches = max(float(epoch_accum['n_batches']), 1.0)
            avg_recon = epoch_accum['recon_sum'] / n_batches
            avg_kl = epoch_accum['kl_sum'] / n_batches
            avg_total = epoch_accum['total_sum'] / n_batches
            loss_history['recon_loss'].append(avg_recon)
            loss_history['KL_loss'].append(avg_kl)
            loss_history['total_loss'].append(avg_total)

            epoch_time = time.perf_counter() - epoch_start
            gpu_mem = torch.cuda.max_memory_allocated(local_device) / 1024**3
            print(f"Epoch {epoch:4d} 완료 |"
                  f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_total:.4f} |\n"
                  f"epoch time : {epoch_time/60:.2f}m | "
                  f"GPU mem: {gpu_mem:.2f}GB |")

            epoch_accum['recon_sum'] = 0.0
            epoch_accum['kl_sum'] = 0.0
            epoch_accum['total_sum'] = 0.0
            epoch_accum['n_batches'] = 0.0

            if epoch % 10 == 0:
                save_checkpoint(current_chunks)
                print(f"  중간 저장 완료 (epoch {epoch})")
        dist.barrier()

    try:
        current_epoch_start = time.perf_counter()
        for global_chunk, epoch, chunk_pos, ci in planned_chunks:
            if chunk_pos == 0:
                current_epoch_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(local_device)

            dataset = _load_chunk_dataset(
                chunk_paths[ci],
                etas,
                eta_min,
                eta_max,
                rank,
                epoch,
                chunk_pos,
                total_chunks,
                ci,
            )

            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed + epoch * total_chunks + chunk_pos,
                drop_last=True,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                pin_memory=True,
                drop_last=True,
            )

            _ddp_log(rank, f"epoch={epoch} chunk_pos={chunk_pos} dataloader ready batches={len(dataloader)}")

            cvae.train()
            bn_mode = apply_bn_mode(global_chunk)
            beta_eff = beta_eff_for_chunk(global_chunk)
            chunk_recon = 0.0
            chunk_kl = 0.0
            chunk_total = 0.0
            chunk_batches = 0
            try:
                for x_batch, eta_batch in dataloader:
                    x_batch = x_batch.to(local_device, non_blocking=True)
                    eta_batch = eta_batch.to(local_device, non_blocking=True)

                    recon_loss, kl_loss = cvae(x_batch, eta_batch)
                    loss = recon_loss + beta_eff * kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
                    optimizer.step()

                    recon_value = recon_loss.item()
                    kl_value = kl_loss.item()
                    total_value = loss.item()
                    chunk_recon += recon_value
                    chunk_kl += kl_value
                    chunk_total += total_value
                    chunk_batches += 1
            except Exception:
                _ddp_log(rank, f"FAILED at epoch={epoch}, chunk_pos={chunk_pos}, ci={ci}")
                traceback.print_exc()
                raise

            chunk_totals = torch.tensor([chunk_recon, chunk_kl, chunk_total, chunk_batches], device=local_device, dtype=torch.float64)
            dist.all_reduce(chunk_totals, op=dist.ReduceOp.SUM)
            chunk_total_batches = max(chunk_totals[3].item(), 1.0)
            chunk_avg_recon = chunk_totals[0].item() / chunk_total_batches
            chunk_avg_kl = chunk_totals[1].item() / chunk_total_batches
            chunk_avg_total = chunk_totals[2].item() / chunk_total_batches

            if rank == 0:
                epoch_accum['recon_sum'] += chunk_totals[0].item()
                epoch_accum['kl_sum'] += chunk_totals[1].item()
                epoch_accum['total_sum'] += chunk_totals[2].item()
                epoch_accum['n_batches'] += chunk_totals[3].item()

            record_chunk_loss(global_chunk, epoch, chunk_pos, ci, (chunk_avg_recon, chunk_avg_kl, chunk_avg_total, beta_eff), bn_mode)
            _ddp_log(rank, f"epoch={epoch} chunk_pos={chunk_pos} done")

            del dataloader, sampler, dataset

            current_chunks = global_chunk + 1
            if current_chunks % total_chunks == 0:
                finish_epoch(epoch, current_epoch_start, current_chunks)

        if rank == 0:
            total_time = time.perf_counter() - train_start
            save_checkpoint(target_chunks)
            print(f"\n모델 저장 완료: {save_path}")
            print(f"\n총 학습 시간: {total_time/60:.2f}분 ({total_time/3600:.2f}시간)")

        dist.barrier()
        return cvae.module, loss_history, eta_min, eta_max
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()




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





