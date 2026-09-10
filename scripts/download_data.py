"""Script to download Flickr8k or create a synthetic sample dataset for offline development."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.data.downloader import download_flickr8k, create_mock_flickr8k_dataset
from src.utils.logger import get_logger

logger = get_logger("download_script")


def main():
    parser = argparse.ArgumentParser(description="Download or generate Flickr8k dataset.")
    parser.add_argument("--mock-only", action="store_true", help="Force generating synthetic dataset without external download")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of synthetic samples to generate if mock mode is used")
    args = parser.parse_args()

    cfg = get_default_config()

    if args.mock_only or not cfg.paths.captions_file.exists():
        logger.info("Initializing dataset setup...")
        if not args.mock_only:
            try:
                download_flickr8k(cfg.paths.raw_data_dir)
            except Exception as e:
                logger.warning(f"Download error: {e}")

        # If captions file still missing, generate mock samples
        if not cfg.paths.captions_file.exists():
            logger.info("Creating mock sample dataset for development...")
            create_mock_flickr8k_dataset(cfg.paths.raw_data_dir, num_samples=args.num_samples)
    else:
        logger.info(f"Dataset already present at {cfg.paths.raw_data_dir}")


if __name__ == "__main__":
    main()
