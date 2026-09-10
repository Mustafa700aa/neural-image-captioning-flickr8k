"""Visualization utilities for Attention Heatmaps, Training Curves, and Qualitative Results."""

import math
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI compatibility
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import pandas as pd


def plot_attention_heatmaps(
    image: Union[str, Path, Image.Image],
    words: List[str],
    alphas: torch.Tensor,
    output_path: Optional[Union[str, Path]] = None,
    encoded_image_size: int = 14,
    smooth: bool = True
) -> plt.Figure:
    """Overlays spatial attention heatmaps onto the image for each generated word token.
    
    Args:
        image: PIL Image or image path
        words: List of generated word tokens
        alphas: Attention weight tensor of shape (seq_len, num_pixels)
        output_path: Optional path to save figure
        encoded_image_size: Spatial grid size (default: 14 for 14x14 = 196)
        smooth: If True, uses bilinear interpolation for smooth heatmap
        
    Returns:
        fig: Matplotlib Figure
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    num_words = len(words)
    if num_words == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis("off")
        return fig

    # Grid layout: num_words + 1 (for original image)
    total_plots = num_words + 1
    cols = min(4, total_plots)
    rows = math.ceil(total_plots / cols)

    fig = plt.figure(figsize=(cols * 4, rows * 4))

    # Plot original image
    ax_orig = fig.add_subplot(rows, cols, 1)
    ax_orig.imshow(img)
    ax_orig.set_title("Input Image", fontsize=12, fontweight="bold")
    ax_orig.axis("off")

    # Plot attention map per word
    for i, word in enumerate(words):
        ax = fig.add_subplot(rows, cols, i + 2)
        ax.imshow(img)

        # Reshape alpha for this word token: (14, 14)
        if i < alphas.size(0):
            alpha_step = alphas[i].detach().cpu().numpy()
            alpha_grid = alpha_step.reshape(encoded_image_size, encoded_image_size)

            interp = "bilinear" if smooth else "nearest"
            ax.imshow(alpha_grid, cmap="jet", alpha=0.6, interpolation=interp, extent=(0, img.width, img.height, 0))

        ax.set_title(f'"{word}"', fontsize=12, fontweight="bold", color="#1a237e")
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_training_history(
    history: Union[str, Path, List[Dict], pd.DataFrame],
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Plots training vs validation loss and perplexity curves across epochs."""
    if isinstance(history, (str, Path)):
        history_path = Path(history)
        if history_path.suffix == ".csv":
            df = pd.read_csv(history_path)
        else:
            df = pd.read_json(history_path)
    elif isinstance(history, list):
        df = pd.DataFrame(history)
    else:
        df = history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    # Loss plot
    axes[0].plot(epochs, df["train_loss"], "o-", label="Train Total Loss", color="#1976D2", lw=2)
    if "val_loss" in df.columns:
        axes[0].plot(epochs, df["val_loss"], "s--", label="Val Total Loss", color="#D32F2F", lw=2)
    if "train_ce_loss" in df.columns:
        axes[0].plot(epochs, df["train_ce_loss"], ":", label="Train Cross-Entropy", color="#0288D1", alpha=0.7)
    axes[0].set_title("Loss Curves", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Perplexity plot
    if "train_perplexity" in df.columns:
        axes[1].plot(epochs, df["train_perplexity"], "o-", label="Train Perplexity", color="#388E3C", lw=2)
    if "val_perplexity" in df.columns:
        axes[1].plot(epochs, df["val_perplexity"], "s--", label="Val Perplexity", color="#F57C00", lw=2)
    axes[1].set_title("Perplexity Curves", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Perplexity (PPL)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig
