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


def _load_chunk_dataset(chunk_path, etas, eta_min, eta_max, barr_type, rank, epoch, chunk_pos, n_chunks, ci):
    _ddp_log(rank, f"epoch={epoch} chunk_pos={chunk_pos}/{n_chunks-1} ci={ci} load start")
    dataset = ChunkDataset(chunk_path, etas, eta_min, eta_max, barr_type)
    _ddp_log(rank, f"epoch={epoch} chunk_pos={chunk_pos} dataset loaded len={len(dataset)}")
    return dataset


def train_chunk_ddp(model_type='hes', barr_type='barr', dim_z=8, hidden_dims=None, batch_size=1024,
                    lr=1e-3, beta=1.0, save_path='cvae_ddp.pt',
                    use_bn=False, num_chunks=None, shuffle_chunks=False, num_workers=4, prefetch_factor=2, seed=1234, overwrite=False,
                    load_path=None, resume_path=None):
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

    dim_x = 2 if barr_type == 'barr' else 1

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

    save_path = os.path.expanduser(save_path)
    resume_path = os.path.expanduser(resume_path) if resume_path is not None else None
    load_path = os.path.expanduser(load_path) if load_path is not None else None
    if os.path.exists(save_path) and not overwrite and save_path != resume_path:
        raise FileExistsError(
            f"{save_path} already exists. Use overwrite=True or choose a different save_path."
        )

    etas, eta_min, eta_max = compute_eta_stats(os.path.expanduser(eta_path))
    dim_eta = etas.shape[1]

    resume_checkpoint = None
    load_model = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=local_device, weights_only=False)
    elif load_path is not None:
        load_model = torch.load(load_path, map_location=local_device, weights_only=False)

    checkpoint = resume_checkpoint or load_model
    if checkpoint is not None:
        checkpoint_use_bn = bool(checkpoint.get('use_bn', False))
        if checkpoint_use_bn != bool(use_bn):
            _rank0_print(rank, f"use_bn을 checkpoint 설정({checkpoint_use_bn})으로 맞춥니다.")
        use_bn = checkpoint_use_bn
    else:
        use_bn = bool(use_bn)

    cvae = CVAE(dim_x=dim_x, dim_eta=dim_eta, dim_z=dim_z, hidden_dims=hidden_dims, use_bn=use_bn).to(local_device)

    completed_chunks = 0
    epoch_accum = {'recon_sum': 0.0, 'kl_sum': 0.0, 'n_batches': 0.0}
    empty_chunk_history = {
        'epoch': [], 'global_chunk': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [],
        'recon_loss': [], 'KL_loss': [], 'total_loss': []
    }

    if resume_checkpoint is not None:
        _load_plain_state_dict(cvae, resume_checkpoint['model_state'])
        completed_chunks = int(resume_checkpoint.get('trained_chunks', len(resume_checkpoint.get('chunk_loss_history', {}).get('chunk_idx', []))))
        epoch_accum = resume_checkpoint.get('epoch_accum', epoch_accum)
        _rank0_print(rank, f"체크포인트 재개: {resume_path} | 완료 chunks={completed_chunks}")
    elif load_model is not None:
        _load_plain_state_dict(cvae, load_model['model_state'])
        completed_chunks = int(load_model.get('trained_chunks', len(load_model.get('chunk_loss_history', {}).get('chunk_idx', []))))
        epoch_accum = load_model.get('epoch_accum', epoch_accum)
        _rank0_print(rank, f"초기 가중치 로드 완료: {load_path} | 완료 chunks={completed_chunks}")

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
    elif load_model is not None:
        loss_history = load_model.get('loss_history', {'recon_loss': [], 'KL_loss': [], 'total_loss': []})
        chunk_loss_history = load_model.get('chunk_loss_history', empty_chunk_history.copy())
    else:
        loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}
        chunk_loss_history = empty_chunk_history.copy()

    if rank == 0:
        chunk_loss_history.setdefault('global_chunk', [])
        if len(chunk_loss_history['global_chunk']) < len(chunk_loss_history.get('chunk_idx', [])):
            chunk_loss_history['global_chunk'] = list(range(len(chunk_loss_history.get('chunk_idx', []))))

    train_start = time.perf_counter()
    run_start_chunk = completed_chunks
    target_chunks = completed_chunks + num_chunks
    _rank0_print(
        rank,
        f"DDP 학습 시작 | world_size={world_size} | per_gpu_batch={batch_size} | "
        f"global_batch={batch_size * world_size} | 이번 실행 chunks={num_chunks} | "
        f"진행 chunks={run_start_chunk}->{target_chunks} | files/epoch={total_chunks}"
    )

    def chunk_order_for_epoch(epoch):
        if shuffle_chunks:
            rng = np.random.default_rng(seed + epoch)
            return rng.permutation(total_chunks)
        return np.arange(total_chunks)

    def chunk_info(global_chunk):
        epoch = global_chunk // total_chunks + 1
        chunk_pos = global_chunk % total_chunks
        chunk_order = chunk_order_for_epoch(epoch)
        ci = int(chunk_order[chunk_pos])
        return epoch, chunk_pos, ci

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

    def record_chunk_loss(global_chunk, epoch, chunk_pos, ci, chunk_losses):
        if rank != 0:
            return
        chunk_avg_recon, chunk_avg_kl, chunk_avg_total = chunk_losses
        chunk_loss_history['epoch'].append(epoch)
        chunk_loss_history['global_chunk'].append(global_chunk)
        chunk_loss_history['chunk_pos'].append(chunk_pos)
        chunk_loss_history['chunk_idx'].append(ci)
        chunk_loss_history['chunk_file'].append(os.path.basename(chunk_paths[ci]))
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
        scheduler.step()
        if rank == 0:
            n_batches = max(float(epoch_accum['n_batches']), 1.0)
            avg_recon = epoch_accum['recon_sum'] / n_batches
            avg_kl = epoch_accum['kl_sum'] / n_batches
            avg_total = avg_recon + beta * avg_kl
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
            epoch_accum['n_batches'] = 0.0

            if epoch % 10 == 0:
                save_checkpoint(current_chunks)
                print(f"  중간 저장 완료 (epoch {epoch})")
        dist.barrier()

    try:
        current_epoch_start = time.perf_counter()
        for offset in range(num_chunks):
            global_chunk = completed_chunks + offset
            epoch, chunk_pos, ci = chunk_info(global_chunk)
            if chunk_pos == 0:
                current_epoch_start = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(local_device)

            dataset = _load_chunk_dataset(
                chunk_paths[ci],
                etas,
                eta_min,
                eta_max,
                barr_type,
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
            chunk_recon = 0.0
            chunk_kl = 0.0
            chunk_batches = 0
            try:
                for x_batch, eta_batch in dataloader:
                    x_batch = x_batch.to(local_device, non_blocking=True)
                    eta_batch = eta_batch.to(local_device, non_blocking=True)

                    recon_loss, kl_loss = cvae(x_batch, eta_batch)
                    loss = recon_loss + beta * kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
                    optimizer.step()

                    recon_value = recon_loss.item()
                    kl_value = kl_loss.item()
                    chunk_recon += recon_value
                    chunk_kl += kl_value
                    chunk_batches += 1
            except Exception:
                _ddp_log(rank, f"FAILED at epoch={epoch}, chunk_pos={chunk_pos}, ci={ci}")
                traceback.print_exc()
                raise

            chunk_totals = torch.tensor([chunk_recon, chunk_kl, chunk_batches], device=local_device, dtype=torch.float64)
            dist.all_reduce(chunk_totals, op=dist.ReduceOp.SUM)
            chunk_total_batches = max(chunk_totals[2].item(), 1.0)
            chunk_avg_recon = chunk_totals[0].item() / chunk_total_batches
            chunk_avg_kl = chunk_totals[1].item() / chunk_total_batches
            chunk_avg_total = chunk_avg_recon + beta * chunk_avg_kl

            if rank == 0:
                epoch_accum['recon_sum'] += chunk_totals[0].item()
                epoch_accum['kl_sum'] += chunk_totals[1].item()
                epoch_accum['n_batches'] += chunk_totals[2].item()

            record_chunk_loss(global_chunk, epoch, chunk_pos, ci, (chunk_avg_recon, chunk_avg_kl, chunk_avg_total))
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





