import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_chunk_loss_history(save_path, show_kl=True, save_fig_path=None):
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    hist = ckpt.get("chunk_loss_history")
    if hist is None:
        raise KeyError("checkpoint does not contain 'chunk_loss_history'")
    if len(hist.get("total_loss", [])) == 0:
        raise ValueError("chunk_loss_history is empty")

    steps = np.arange(1, len(hist["total_loss"]) + 1)
    epochs = np.asarray(hist.get("epoch", []))

    plt.figure(figsize=(9, 5))
    plt.plot(steps, hist["recon_loss"], label="Recon", linewidth=1.8)
    if show_kl:
        plt.plot(steps, hist["KL_loss"], label="KL", linewidth=1.4)
    plt.plot(steps, hist["total_loss"], label="Total", linewidth=1.8)

    if len(epochs) == len(steps):
        boundaries = np.where(epochs[1:] != epochs[:-1])[0] + 1
        for boundary in boundaries:
            plt.axvline(boundary + 0.5, color="gray", linestyle="--", alpha=0.35, linewidth=1)

    plt.xlabel("Chunk step")
    plt.ylabel("Loss")
    plt.title("Chunk-level loss history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    return hist


def plot_epoch_loss_history(save_path, show_kl=True, save_fig_path=None):
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    hist = ckpt.get("loss_history")
    if hist is None:
        raise KeyError("checkpoint does not contain 'loss_history'")
    if len(hist.get("total_loss", [])) == 0:
        raise ValueError("loss_history is empty")

    epochs = np.arange(1, len(hist["total_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, hist["recon_loss"], label="Recon", linewidth=1.8)
    if show_kl:
        plt.plot(epochs, hist["KL_loss"], label="KL", linewidth=1.4)
    plt.plot(epochs, hist["total_loss"], label="Total", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch-level loss history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    return hist
