"""CLI script to push trained Image Captioning checkpoint and vocabulary to HuggingFace Hub."""

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.utils.hub import push_model_to_hub
from src.utils.logger import get_logger

logger = get_logger("hub_script")


def main():
    parser = argparse.ArgumentParser(description="Upload Image Captioning Model to HuggingFace Hub.")
    parser.add_argument("--repo-id", type=str, default="AntigravityAI/image-caption-flickr8k", help="HuggingFace Hub repository ID (e.g. username/repo-name)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to caption_model_best.pt")
    parser.add_argument("--vocab", type=str, default=None, help="Path to vocab.json")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN", None), help="HuggingFace API token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    cfg = get_default_config()
    
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.paths.checkpoints_dir / "caption_model_best.pt"
    vocab_path = Path(args.vocab) if args.vocab else cfg.paths.vocab_path

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path}. Run training first with: python scripts/train.py")
        return

    if not vocab_path.exists():
        logger.error(f"Vocabulary file not found at: {vocab_path}")
        return

    try:
        url = push_model_to_hub(
            repo_id=args.repo_id,
            checkpoint_path=ckpt_path,
            vocab_path=vocab_path,
            token=args.token,
            private=args.private
        )
        print(f"\n🎉 Successfully published model to HuggingFace Hub: {url}\n")
    except Exception as e:
        logger.error(f"Failed to publish to HuggingFace Hub: {e}")


if __name__ == "__main__":
    main()
