import argparse

from e_1_run_cvae_ajoufe1 import train_chunk_ddp


def parse_hidden_dims(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train CVAE with DistributedDataParallel.")
    parser.add_argument("--model-type", choices=["hes", "bs"], default="bs")
    parser.add_argument("--dim-z", type=int, default=8)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[128, 128, 64])
    parser.add_argument("--use-bn", action="store_true", help="Use BatchNorm1d in CVAE hidden layers.")
    parser.add_argument("--bn-chunks", type=int, default=None, help="Keep BatchNorm trainable for this many global chunk steps, then freeze it.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size per GPU.")
    parser.add_argument("--num-chunks", type=int, default=None, help="Train this many chunk steps from the saved progress position.")
    parser.add_argument("--shuffle-chunks", dest="shuffle_chunks", action="store_true", default=True, help="Shuffle chunk order within each epoch. Enabled by default.")
    parser.add_argument("--no-shuffle-chunks", dest="shuffle_chunks", action="store_false", help="Read chunk files in sorted order.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--warmup-chunks", type=int, default=None, help="Linearly warm up KL beta over this many global chunk steps.")
    parser.add_argument("--save-path", default=None, help="Optional override for the auto-generated checkpoint path.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU process.")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite save-path if it already exists.")
    parser.add_argument("--resume-path", default=None, help="Checkpoint to resume model, optimizer, scheduler, epoch, and loss history from.")
    args = parser.parse_args()

    train_chunk_ddp(
        model_type=args.model_type,
        dim_z=args.dim_z,
        hidden_dims=args.hidden_dims,
        use_bn=args.use_bn,
        bn_chunks=args.bn_chunks,
        batch_size=args.batch_size,
        num_chunks=args.num_chunks,
        shuffle_chunks=args.shuffle_chunks,
        lr=args.lr,
        beta=args.beta,
        warmup_chunks=args.warmup_chunks,
        save_path=args.save_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        overwrite=args.overwrite,
        resume_path=args.resume_path,
    )


if __name__ == "__main__":
    main()
