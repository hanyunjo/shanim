import argparse

from e_1_run_cvae import train_chunk_ddp


def parse_hidden_dims(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train CVAE with DistributedDataParallel.")
    parser.add_argument("--model-type", choices=["hes", "bs"], default="hes")
    parser.add_argument("--barr-type", choices=["barr", "van"], default="barr")
    parser.add_argument("--dim-z", type=int, default=8)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[128, 128, 64])
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--save-path", default="cvae_ddp.pt")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU process.")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite save-path if it already exists.")
    parser.add_argument("--load-path", default=None, help="Optional checkpoint to initialize from.")
    args = parser.parse_args()

    train_chunk_ddp(
        model_type=args.model_type,
        barr_type=args.barr_type,
        dim_z=args.dim_z,
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        save_path=args.save_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        overwrite=args.overwrite,
        load_path=args.load_path,
    )


if __name__ == "__main__":
    main()
