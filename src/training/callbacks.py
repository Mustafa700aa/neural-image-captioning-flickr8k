"""Training callbacks: Early Stopping, Checkpointing, and Metric Logging."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger("callbacks")


class EarlyStopping:
    """Early stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")

        return self.early_stop


class ModelCheckpoint:
    """Saves model weights, optimizer state, and training metadata."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        mode: str = "min",
        save_top_k: int = 3,
        filename_prefix: str = "caption_model"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_prefix = filename_prefix
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: List[Dict[str, Any]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_metric: float,
        metrics: Dict[str, float],
        config: Any,
        is_best: bool = False
    ) -> Path:
        """Saves a checkpoint file."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metric": val_metric,
            "metrics": metrics,
            "config": config
        }

        # Checkpoint path for this epoch
        ckpt_path = self.checkpoint_dir / f"{self.filename_prefix}_epoch_{epoch:02d}.pt"
        torch.save(state, ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.filename_prefix}_best.pt"
            torch.save(state, best_path)
            logger.info(f"Saved new best model checkpoint to {best_path} (val_metric: {val_metric:.4f})")

        # Save metadata history
        history_file = self.checkpoint_dir / "training_history.json"
        self.saved_checkpoints.append({
            "epoch": epoch,
            "path": str(ckpt_path),
            "val_metric": val_metric,
            "metrics": metrics,
            "is_best": is_best
        })
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.saved_checkpoints, f, indent=2)

        return ckpt_path

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        current_score: float,
        metrics: Dict[str, float],
        config: Any
    ) -> bool:
        """Evaluates score and saves checkpoint, returning True if this epoch was the best so far."""
        if self.mode == "min":
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score

        if is_best:
            self.best_score = current_score

        self.save(model, optimizer, epoch, current_score, metrics, config, is_best=is_best)
        return is_best


class MetricTracker:
    """Tracks and logs training and validation metrics across epochs."""

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    def update(self, epoch: int, metrics: Dict[str, float]) -> None:
        record = {"epoch": epoch, **metrics}
        self.history.append(record)

        # Save CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_dir / "metrics_log.csv", index=False)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
