import time
import numpy as np
import torch
import h5py
import glob
import os
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



# ─────────────
# 2.train
# ─────────────
def train_chunk(model_type = 'hes', barr_type='barr', dim_z=8, hidden_dims=None, batch_size=1024,
                n_epochs=200, lr=1e-3, beta=1.0, use_bn=False, num_chunks=None,
                shuffle_chunks=True, save_path='cvae.pt', load_path=None, resume_path=None):
    
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

    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.h5")))
    total_chunks = len(chunk_paths)
    if total_chunks == 0:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    if num_chunks is not None:
        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        if num_chunks > total_chunks:
            raise ValueError(f"num_chunks={num_chunks} is larger than available chunks={total_chunks}")
        chunk_paths = chunk_paths[:num_chunks]
    n_chunks = len(chunk_paths)

    save_path = os.path.expanduser(save_path)
    load_path = os.path.expanduser(load_path) if load_path is not None else None
    resume_path = os.path.expanduser(resume_path) if resume_path is not None else None

    # 기존 모델 확인
    if load_path is None and resume_path is None and os.path.exists(save_path):
        choice = input(f"{save_path} 존재합니다. (1:불러오기 / 2:덮어쓰기 / 3:중지): ")

        if choice == '1':
            load_model = torch.load(save_path, map_location=device, weights_only=False)
            cvae = CVAE(dim_x=load_model['dim_x'], 
                        dim_eta=load_model['dim_eta'],
                        dim_z=load_model['dim_z'], 
                        hidden_dims=load_model['hidden_dims'],
                        use_bn=load_model.get('use_bn', False)
                        )
            cvae.load_state_dict(load_model['model_state'])
            cvae.to(device)
            cvae.eval()
            print("모델 불러오기 완료")
            return cvae, load_model['loss_history'], load_model['eta_min'], load_model['eta_max']

        elif choice == '2':
            print("덮어쓰기 학습")
        elif choice == '3':
            raise ValueError("중지")

    etas, eta_min, eta_max = compute_eta_stats(eta_path)
    dim_eta = etas.shape[1]

    resume_checkpoint = None
    load_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    elif load_path is not None:
        load_checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    checkpoint = resume_checkpoint or load_checkpoint
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

    start_epoch = 0
    if resume_path is not None: # epoch을 이어서 학습하기 위해
        ckpt = resume_checkpoint
        cvae.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        loss_history = ckpt["loss_history"]
        chunk_loss_history = ckpt.get('chunk_loss_history', {'epoch': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [], 'recon_loss': [], 'KL_loss': [], 'total_loss': []})
        eta_min = ckpt["eta_min"]
        eta_max = ckpt["eta_max"]
        start_epoch = int(ckpt["epoch"])
        print(f"체크포인트 재개: {resume_path} | 완료 epoch={start_epoch}")
    elif load_path is not None: # 모델 weight만 불러와 새 optimizer/scheduler로 추가 학습
        load_model = load_checkpoint
        cvae.load_state_dict(load_model["model_state"])
        eta_min = load_model["eta_min"]
        eta_max = load_model["eta_max"]
        loss_history = load_model.get('loss_history', {'recon_loss': [], 'KL_loss': [], 'total_loss': []})
        chunk_loss_history = load_model.get('chunk_loss_history', {'epoch': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [], 'recon_loss': [], 'KL_loss': [], 'total_loss': []})
        start_epoch = int(load_model.get("epoch", len(loss_history.get('total_loss', []))))
        print(f"초기 가중치 로드 완료: {load_path} | 완료 epoch={start_epoch}")
    else:
        loss_history = {'recon_loss': [], 'KL_loss': [], 'total_loss': []}
        chunk_loss_history = {'epoch': [], 'chunk_pos': [], 'chunk_idx': [], 'chunk_file': [], 'recon_loss': [], 'KL_loss': [], 'total_loss': []}


    train_start = time.perf_counter()
    epoch_times = []
    target_epoch = start_epoch + n_epochs
    print(f"학습 시작 | chunks={n_chunks}/{total_chunks}")
    for epoch_offset in range(1, n_epochs + 1):
        epoch = start_epoch + epoch_offset
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        epoch_recon = 0.0
        epoch_kl    = 0.0
        n_batches   = 0
        if shuffle_chunks:
            chunk_order = np.random.permutation(n_chunks)   # 청크 순서 셔플
        else:
            chunk_order = np.arange(n_chunks)

        def load_chunk(ci):
            dataset = ChunkDataset(chunk_paths[ci], etas, eta_min, eta_max, barr_type)
            return dataset

        def train_one_chunk(dataset):
            nonlocal epoch_recon, epoch_kl, n_batches

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                persistent_workers=True,
                # DataLoader가 worker 프로세스를 유지하여 다음 epoch에서도 재사용.
                # 매 epoch마다 worker 프로세스 생성/종료하는 오버헤드 감소.
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
                epoch_recon += recon_value
                epoch_kl += kl_value
                n_batches += 1
                chunk_recon += recon_value
                chunk_kl += kl_value
                chunk_batches += 1

            del dataloader

            chunk_batches = max(chunk_batches, 1)
            chunk_avg_recon = chunk_recon / chunk_batches
            chunk_avg_kl = chunk_kl / chunk_batches
            chunk_avg_total = chunk_avg_recon + beta * chunk_avg_kl
            return chunk_avg_recon, chunk_avg_kl, chunk_avg_total

        def record_chunk_loss(chunk_pos, ci, chunk_losses):
            chunk_avg_recon, chunk_avg_kl, chunk_avg_total = chunk_losses
            chunk_loss_history['epoch'].append(epoch)
            chunk_loss_history['chunk_pos'].append(chunk_pos)
            chunk_loss_history['chunk_idx'].append(int(ci))
            chunk_loss_history['chunk_file'].append(os.path.basename(chunk_paths[int(ci)]))
            chunk_loss_history['recon_loss'].append(chunk_avg_recon)
            chunk_loss_history['KL_loss'].append(chunk_avg_kl)
            chunk_loss_history['total_loss'].append(chunk_avg_total)
            print(
                f"Chunk {chunk_pos + 1:3d}/{n_chunks} | epoch {epoch:4d} | "
                f"Recon: {chunk_avg_recon:.4f} | KL: {chunk_avg_kl:.4f} | Total: {chunk_avg_total:.4f}",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=1) as prefetcher:
            future = prefetcher.submit(load_chunk, chunk_order[0])

            for chunk_pos, next_ci in enumerate(chunk_order[1:]):
                ci = int(chunk_order[chunk_pos])
                dataset = future.result()
                future = prefetcher.submit(load_chunk, int(next_ci))

                chunk_losses = train_one_chunk(dataset)
                record_chunk_loss(chunk_pos, ci, chunk_losses)

                del dataset

            last_chunk_pos = n_chunks - 1
            last_ci = int(chunk_order[-1])
            dataset = future.result() # last chunk training
            chunk_losses = train_one_chunk(dataset)
            record_chunk_loss(last_chunk_pos, last_ci, chunk_losses)
            del dataset

        scheduler.step()

        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches
        avg_total = avg_recon + beta * avg_kl

        loss_history['recon_loss'].append(avg_recon)
        loss_history['KL_loss'].append(avg_kl)
        loss_history['total_loss'].append(avg_total)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_epoch_time * (n_epochs - epoch_offset)
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{target_epoch} |\n"
                  f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_total:.4f} |\n"
                  f"epoch time : {epoch_time/60:.2f}m | remaining time: {remaining/60:.2f}m |\n"
                  f"GPU mem: {gpu_mem:.2f}GB"
                  )

        # 10 epoch마다 중간 저장
        if epoch % 10 == 0:
            torch.save({
                'model_state' : cvae.state_dict(), # model weight
                'eta_min'     : eta_min,
                'eta_max'     : eta_max,
                'dim_x'       : dim_x,
                'dim_eta'     : dim_eta,
                'dim_z'       : dim_z,
                'hidden_dims' : hidden_dims,
                'use_bn'      : use_bn,
                'loss_history': loss_history,
                'chunk_loss_history': chunk_loss_history,
                'num_chunks'  : n_chunks,
                'total_chunks': total_chunks,
                'shuffle_chunks': shuffle_chunks,
                'epoch'          : epoch,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
            }, save_path)
            print(f"  중간 저장 완료 (epoch {epoch})")

    total_time = time.perf_counter() - train_start
    torch.save({
        'model_state' : cvae.state_dict(),
        'eta_min'     : eta_min,
        'eta_max'     : eta_max,
        'dim_x'       : dim_x,
        'dim_eta'     : dim_eta,
        'dim_z'       : dim_z,
        'hidden_dims' : hidden_dims,
        'use_bn'      : use_bn,
        'loss_history': loss_history,
        'chunk_loss_history': chunk_loss_history,
        'num_chunks'  : n_chunks,
        'total_chunks': total_chunks,
        'shuffle_chunks': shuffle_chunks,
        'epoch'          : target_epoch,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }, save_path)
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





