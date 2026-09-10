"""Training script for Image Captioning model."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from src.config import get_default_config
from src.data.dataset import create_dataloaders
from src.data.downloader import create_mock_flickr8k_dataset
from src.models.captioner import ImageCaptionModel
from src.training.trainer import CaptionTrainer
from src.evaluation.visualization import plot_training_history
from src.utils.device import set_seed, get_device
from src.utils.logger import get_logger

logger = get_logger("train_script")


def main():
    parser = argparse.ArgumentParser(description="Train Neural Image Caption Generator.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--backbone", type=str, default="resnet50", help="CNN backbone")
    parser.add_argument("--decoder-lr", type=float, default=4e-4, help="Decoder learning rate")
    parser.add_argument("--encoder-lr", type=float, default=1e-4, help="Encoder learning rate")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune CNN encoder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Optionally limit dataset samples for fast training")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Resume training from a checkpoint path, or 'latest' to automatically detect the latest saved epoch checkpoint."
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.training.num_epochs = args.epochs
    cfg.data.batch_size = args.batch_size
    cfg.model.encoder_name = args.backbone
    cfg.model.decoder_lr = args.decoder_lr
    cfg.model.encoder_lr = args.encoder_lr
    cfg.model.fine_tune_encoder = args.fine_tune
    cfg.training.seed = args.seed

    set_seed(cfg.training.seed)
    device = get_device()

    # Ensure data exists; if not, create mock data for immediate training/testing
    if not cfg.paths.captions_file.exists():
        logger.warning(f"Captions file missing at {cfg.paths.captions_file}. Creating sample dataset...")
        create_mock_flickr8k_dataset(cfg.paths.raw_data_dir, num_samples=60)

    # If max_samples specified, load and slice dataframe
    df = None
    if args.max_samples is not None:
        from src.data.dataset import parse_flickr8k_captions
        full_df = parse_flickr8k_captions(cfg.paths.captions_file)
        unique_imgs = full_df["image_name"].drop_duplicates().sample(n=min(args.max_samples, full_df["image_name"].nunique()), random_state=args.seed)
        df = full_df[full_df["image_name"].isin(unique_imgs)].reset_index(drop=True)
        logger.info(f"Subsampled dataset to {len(unique_imgs)} images ({len(df)} captions) for fast execution.")

    logger.info("Building dataset loaders and vocabulary...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(cfg, df=df)
    logger.info(f"Vocabulary size: {len(vocab)} words. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Adjust encoder_dim based on backbone
    if args.backbone == "mobilenet_v3_large":
        cfg.model.encoder_dim = 960
    elif args.backbone == "efficientnet_b0":
        cfg.model.encoder_dim = 1280
    else:
        cfg.model.encoder_dim = 2048

    cfg.model.vocab_size = len(vocab)

    model = ImageCaptionModel(
        config=cfg.model,
        vocab_size=len(vocab),
        encoder_dim=cfg.model.encoder_dim,
        decoder_dim=cfg.model.decoder_dim,
        attention_dim=cfg.model.attention_dim,
        embed_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
        backbone=cfg.model.encoder_name,
        fine_tune_encoder=cfg.model.fine_tune_encoder
    )

    # Check resume argument
    resume_path = None
    if args.resume is not None:
        if args.resume.lower() == "latest":
            # Search for latest checkpoint in checkpoints_dir
            ckpt_files = sorted(
                list(cfg.paths.checkpoints_dir.glob("caption_model_epoch_*.pt")),
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
            )
            if ckpt_files:
                resume_path = ckpt_files[-1]
                logger.info(f"Auto-detected latest checkpoint: {resume_path}")
            elif (cfg.paths.checkpoints_dir / "caption_model_best.pt").exists():
                resume_path = cfg.paths.checkpoints_dir / "caption_model_best.pt"
                logger.info(f"Using best checkpoint: {resume_path}")
            else:
                logger.warning("No checkpoint found in checkpoints directory. Starting from epoch 1.")
        else:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                logger.error(f"Specified resume checkpoint does not exist: {resume_path}")
                sys.exit(1)

    trainer = CaptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        pad_idx=vocab.pad_idx
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    results = trainer.fit()
    logger.info(f"Training finished. Best validation loss: {results['best_val_loss']:.4f}")

    # Plot loss and perplexity curves
    metrics_csv = cfg.paths.outputs_dir / "metrics_log.csv"
    if metrics_csv.exists():
        plot_training_history(metrics_csv, output_path=cfg.paths.outputs_dir / "training_curves.png")
        logger.info(f"Saved training curves to {cfg.paths.outputs_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
