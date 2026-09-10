import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.captioner import ImageCaptionModel
from src.models.loss import CaptionLoss
from src.training.callbacks import EarlyStopping, ModelCheckpoint, MetricTracker
from src.config import GlobalConfig, get_default_config
from src.utils.logger import get_logger
from src.utils.device import get_device

logger = get_logger("trainer")


class CaptionTrainer:
    """Orchestrates model training, validation, optimization, and checkpointing."""

    def __init__(
        self,
        model: ImageCaptionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[GlobalConfig] = None,
        device: Optional[torch.device] = None,
        pad_idx: int = 0
    ):
        self.cfg = config or get_default_config()
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_idx = pad_idx
        self.start_epoch = 1

        # Loss function
        self.criterion = CaptionLoss(pad_idx=self.pad_idx, alpha_c=self.cfg.training.alpha_c).to(self.device)

        # Setup Optimizer with parameter groups (encoder fine-tuning vs decoder)
        params_to_opt = [
            {"params": self.model.decoder.parameters(), "lr": self.cfg.model.decoder_lr, "weight_decay": self.cfg.training.weight_decay}
        ]
        if self.cfg.model.fine_tune_encoder:
            params_to_opt.append({
                "params": filter(lambda p: p.requires_grad, self.model.encoder.parameters()),
                "lr": self.cfg.model.encoder_lr,
                "weight_decay": self.cfg.training.weight_decay
            })

        self.optimizer = torch.optim.AdamW(params_to_opt)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.training.lr_scheduler_factor,
            patience=self.cfg.training.lr_scheduler_patience,
            min_lr=1e-6
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=self.cfg.training.patience,
            min_delta=self.cfg.training.min_delta,
            mode="min"
        )
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=self.cfg.paths.checkpoints_dir,
            mode="min",
            save_top_k=self.cfg.training.save_top_k
        )
        self.metric_tracker = MetricTracker(log_dir=self.cfg.paths.outputs_dir)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Loads model weights, optimizer state, and training metadata to resume training.

        Args:
            checkpoint_path: Path to the saved checkpoint (.pt file).

        Returns:
            The starting epoch for training resumption.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        logger.info(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # 1. Restore Model Weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint, strict=False)

        # 2. Restore Optimizer State
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Restored optimizer state successfully.")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")

        # 3. Determine resume epoch
        last_epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
        self.start_epoch = last_epoch + 1
        logger.info(f"Resuming training from epoch {self.start_epoch} (completed epochs: {last_epoch})")

        # 4. Restore history and best metric in callbacks
        history_file = self.cfg.paths.checkpoints_dir / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                valid_history = [h for h in history if h.get("epoch", 0) <= last_epoch]
                self.checkpoint.saved_checkpoints = valid_history
                if valid_history:
                    best_score = min(h.get("val_metric", float("inf")) for h in valid_history)
                    self.checkpoint.best_score = best_score
                    self.early_stopping.best_score = best_score

                    # Calculate early stopping counter
                    counter = 0
                    for h in reversed(valid_history):
                        if abs(h.get("val_metric", float("inf")) - best_score) < 1e-6:
                            break
                        counter += 1
                    self.early_stopping.counter = counter
                    logger.info(f"Restored best validation loss: {best_score:.4f} (EarlyStopping counter: {counter}/{self.early_stopping.patience})")
            except Exception as e:
                logger.warning(f"Could not restore checkpoint history: {e}")

        # 5. Restore MetricTracker history
        metrics_csv = self.cfg.paths.outputs_dir / "metrics_log.csv"
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                df = df[df["epoch"] <= last_epoch]
                self.metric_tracker.history = df.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Could not restore metric tracker history: {e}")

        return self.start_epoch

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch over the training dataset."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_reg_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.training.num_epochs} [Train]")
        for batch_idx, (images, captions, lengths, _) in enumerate(pbar):
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            scores, caps_sorted, decode_lengths, alphas = self.model(images, captions, lengths)
            loss, ce_loss, reg_loss = self.criterion(scores, caps_sorted, decode_lengths, alphas)

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            if self.cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_reg_loss += reg_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "reg": f"{reg_loss.item():.4f}"
            })

        avg_loss = total_loss / max(1, num_batches)
        avg_ce = total_ce_loss / max(1, num_batches)
        avg_reg = total_reg_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_ce, 20.0))

        return {
            "train_loss": avg_loss,
            "train_ce_loss": avg_ce,
            "train_reg_loss": avg_reg,
            "train_perplexity": perplexity
        }

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluates model performance on the validation dataset."""
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_reg_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.cfg.training.num_epochs} [Val]")
            for images, captions, lengths, _ in pbar:
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)

                scores, caps_sorted, decode_lengths, alphas = self.model(images, captions, lengths)
                loss, ce_loss, reg_loss = self.criterion(scores, caps_sorted, decode_lengths, alphas)

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_reg_loss += reg_loss.item()

        avg_loss = total_loss / max(1, num_batches)
        avg_ce = total_ce_loss / max(1, num_batches)
        avg_reg = total_reg_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_ce, 20.0))

        return {
            "val_loss": avg_loss,
            "val_ce_loss": avg_ce,
            "val_reg_loss": avg_reg,
            "val_perplexity": perplexity
        }

    def fit(self) -> Dict[str, Any]:
        """Executes full training loop across all configured epochs."""
        if self.start_epoch > self.cfg.training.num_epochs:
            logger.info(
                f"Model has already been trained for {self.start_epoch - 1} epochs. "
                f"Target epochs is {self.cfg.training.num_epochs}. Nothing to train."
            )
            return {
                "best_val_loss": self.checkpoint.best_score,
                "total_time_sec": 0,
                "history": self.metric_tracker.get_history()
            }

        logger.info(
            f"Starting training on device: {self.device} from epoch {self.start_epoch} "
            f"to {self.cfg.training.num_epochs}."
        )
        start_time = time.time()
        best_val_loss = self.checkpoint.best_score if self.checkpoint.best_score != float("inf") else float("inf")

        for epoch in range(self.start_epoch, self.cfg.training.num_epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            epoch_time = time.time() - epoch_start
            combined_metrics = {**train_metrics, **val_metrics, "epoch_time_sec": epoch_time}

            # Update learning rate scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["val_loss"])
            combined_metrics["learning_rate"] = current_lr

            # Log metrics
            self.metric_tracker.update(epoch, combined_metrics)
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_metrics['train_loss']:.4f} (PPL: {train_metrics['train_perplexity']:.2f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} (PPL: {val_metrics['val_perplexity']:.2f}) | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # Checkpoint step
            is_best = self.checkpoint.step(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                current_score=val_metrics["val_loss"],
                metrics=combined_metrics,
                config=self.cfg
            )
            if is_best:
                best_val_loss = val_metrics["val_loss"]

            # Early stopping check
            if self.early_stopping(val_metrics["val_loss"]):
                logger.info(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes. Best Val Loss: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "total_time_sec": total_time,
            "history": self.metric_tracker.get_history()
        }
