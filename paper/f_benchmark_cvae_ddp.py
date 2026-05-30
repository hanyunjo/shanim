import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from e_1_run_cvae import (
    BS_CHUNK_DIR,
    BS_ETA_PATH,
    HES_CHUNK_DIR,
    HES_ETA_PATH,
    ChunkDataset,
)
from e_2_CVAE import CVAE


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_eta_stats(eta_path):
    with h5py.File(os.path.expanduser(eta_path), "r") as f:
        etas = f["etas"][:].astype(np.float32)
    return etas, etas.min(axis=0), etas.max(axis=0)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def resolve_paths(model_type):
    if model_type == "hes":
        return HES_CHUNK_DIR, HES_ETA_PATH
    return BS_CHUNK_DIR, BS_ETA_PATH


def benchmark_batch_size(args, batch_size, rank, local_rank, world_size, device):
    chunk_dir, eta_path = resolve_paths(args.model_type)
    chunk_paths = sorted(glob.glob(os.path.join(os.path.expanduser(chunk_dir), "*.h5")))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    if args.num_chunks > 0:
        chunk_paths = chunk_paths[:args.num_chunks]

    etas, eta_min, eta_max = load_eta_stats(eta_path)
    dim_x = 2 if args.barr_type == "barr" else 1
    dim_eta = etas.shape[1]

    torch.manual_seed(args.seed)
    model = CVAE(
        dim_x=dim_x,
        dim_eta=dim_eta,
        dim_z=args.dim_z,
        hidden_dims=args.hidden_dims,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rank0_print(
        rank,
        f"\n[batch_size={batch_size}] per_gpu={batch_size}, global={batch_size * world_size}, chunks={len(chunk_paths)}",
    )

    total_train_time = 0.0
    total_wall_time = 0.0
    total_batches = 0
    total_recon = 0.0
    total_kl = 0.0

    for chunk_i, chunk_path in enumerate(chunk_paths, start=1):
        wall_start = time.perf_counter()
        dataset = ChunkDataset(chunk_path, etas, eta_min, eta_max, args.barr_type)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed + chunk_i,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=True,
            drop_last=True,
        )

        dist.barrier()
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()

        local_batches = 0
        local_recon = 0.0
        local_kl = 0.0
        model.train()
        for x_batch, eta_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            eta_batch = eta_batch.to(device, non_blocking=True)

            recon_loss, kl_loss = model(x_batch, eta_batch)
            loss = recon_loss + args.beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            local_batches += 1
            local_recon += recon_loss.item()
            local_kl += kl_loss.item()

        torch.cuda.synchronize(device)
        train_time = time.perf_counter() - train_start
        dist.barrier()
        wall_time = time.perf_counter() - wall_start

        stats = torch.tensor(
            [train_time, wall_time, local_batches, local_recon, local_kl],
            device=device,
            dtype=torch.float64,
        )
        dist.reduce(stats, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_train_time = stats[0].item() / world_size
            avg_wall_time = stats[1].item() / world_size
            batches = int(stats[2].item())
            avg_recon = stats[3].item() / max(batches, 1)
            avg_kl = stats[4].item() / max(batches, 1)

            total_train_time += avg_train_time
            total_wall_time += avg_wall_time
            total_batches += batches
            total_recon += stats[3].item()
            total_kl += stats[4].item()

            print(
                f"chunk {chunk_i:3d}/{len(chunk_paths)} | "
                f"train {avg_train_time:8.2f}s | wall {avg_wall_time:8.2f}s | "
                f"batches {batches:6d} | recon {avg_recon:.5f} | kl {avg_kl:.5f}"
            )

        del dataloader, sampler, dataset

    if rank == 0:
        avg_train_per_chunk = total_train_time / len(chunk_paths)
        avg_wall_per_chunk = total_wall_time / len(chunk_paths)
        est_epoch_train_min = avg_train_per_chunk * args.total_chunks / 60
        est_epoch_wall_min = avg_wall_per_chunk * args.total_chunks / 60
        avg_recon_all = total_recon / max(total_batches, 1)
        avg_kl_all = total_kl / max(total_batches, 1)

        print(
            f"SUMMARY batch_size={batch_size} | "
            f"avg_train/chunk={avg_train_per_chunk:.2f}s | "
            f"avg_wall/chunk={avg_wall_per_chunk:.2f}s | "
            f"est_epoch_train={est_epoch_train_min:.2f}m/{args.total_chunks}chunks | "
            f"est_epoch_wall={est_epoch_wall_min:.2f}m/{args.total_chunks}chunks | "
            f"avg_recon={avg_recon_all:.5f} | avg_kl={avg_kl_all:.5f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CVAE DDP training speed for several batch sizes.")
    parser.add_argument("--model-type", choices=["hes", "bs"], default="hes")
    parser.add_argument("--barr-type", choices=["barr", "van"], default="barr")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[512, 1024, 2048])
    parser.add_argument("--num-chunks", type=int, default=3, help="Number of chunk files to benchmark.")
    parser.add_argument("--total-chunks", type=int, default=100, help="Total chunks used for epoch-time estimate.")
    parser.add_argument("--dim-z", type=int, default=8)
    parser.add_argument("--hidden-dims", type=parse_int_list, default=[128, 128, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_ddp()
    try:
        rank0_print(rank, f"DDP benchmark start | world_size={world_size} | device={device}")
        for batch_size in args.batch_sizes:
            benchmark_batch_size(args, batch_size, rank, local_rank, world_size, device)
            dist.barrier()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
