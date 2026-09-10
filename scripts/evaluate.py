"""Evaluation script for computing benchmark metrics on test set."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.data.dataset import create_dataloaders
from src.inference.predictor import CaptionPredictor
from src.evaluation.evaluator import evaluate_dataset
from src.utils.logger import get_logger

logger = get_logger("eval_script")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model on Test Dataset.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--method", type=str, default="beam", choices=["beam", "greedy"], help="Decoding method")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate")
    args = parser.parse_args()

    cfg = get_default_config()
    
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        best_ckpt = cfg.paths.checkpoints_dir / "caption_model_best.pt"
        if best_ckpt.exists():
            ckpt_path = best_ckpt
        else:
            logger.warning(f"No checkpoint specified and default {best_ckpt} not found. Running with initialized weights.")

    logger.info("Loading dataloaders...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(cfg)

    predictor = CaptionPredictor(
        checkpoint_path=ckpt_path,
        vocab=vocab,
        config=cfg
    )

    output_json = cfg.paths.outputs_dir / "evaluation_results.json"
    evaluate_dataset(
        predictor=predictor,
        test_loader=test_loader,
        output_json=output_json,
        method=args.method,
        beam_width=args.beam_width,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
