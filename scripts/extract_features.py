"""Script to pre-extract CNN spatial features to disk."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.training.feature_extractor import extract_and_cache_features
from src.utils.logger import get_logger

logger = get_logger("extract_script")


def main():
    parser = argparse.ArgumentParser(description="Extract CNN spatial features from images.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone model (resnet50, mobilenet_v3_large, efficientnet_b0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for extraction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached features")
    args = parser.parse_args()

    cfg = get_default_config()
    images_dir = cfg.paths.raw_data_dir / "Images"
    features_dir = cfg.paths.features_dir

    if not images_dir.exists():
        logger.error(f"Images directory not found at {images_dir}. Run scripts/download_data.py first.")
        return

    extract_and_cache_features(
        images_dir=images_dir,
        features_dir=features_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
