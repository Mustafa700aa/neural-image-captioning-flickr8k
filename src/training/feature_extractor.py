"""Offline CNN feature extraction and disk caching pipeline."""

from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.encoder import EncoderCNN
from src.data.transforms import get_val_transforms
from src.utils.logger import get_logger
from src.utils.device import get_device

logger = get_logger("feature_extractor")


class ImageListDataset(Dataset):
    """Simple dataset for loading images from disk for batch feature extraction."""

    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, path.name, True
        except Exception as e:
            logger.warning(f"Error loading image {path}: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32), path.name, False


def extract_and_cache_features(
    images_dir: Union[str, Path],
    features_dir: Union[str, Path],
    backbone: str = "resnet50",
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    overwrite: bool = False
) -> int:
    """Extracts spatial visual features for all images in images_dir and saves them as .pt files.
    
    Args:
        images_dir: Directory containing Flickr8k image files.
        features_dir: Target directory where .pt feature tensors are saved.
        backbone: Name of pretrained CNN backbone.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        device: Torch compute device (CPU/CUDA).
        overwrite: If True, re-extracts existing feature files.
        
    Returns:
        count: Total number of features cached.
    """
    images_dir = Path(images_dir)
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()

    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    if not all_images:
        logger.warning(f"No images found in {images_dir}")
        return 0

    # Filter out already extracted images if not overwriting
    if not overwrite:
        images_to_process = []
        for img in all_images:
            feat_file = features_dir / f"{img.stem}.pt"
            if feat_file.exists() and feat_file.stat().st_size > 1000:
                continue
            images_to_process.append(img)
    else:
        images_to_process = all_images

    if not images_to_process:
        logger.info(f"All {len(all_images)} images already have cached features in {features_dir}")
        return len(all_images)

    logger.info(f"Extracting features for {len(images_to_process)} images using {backbone} on {device}...")

    encoder = EncoderCNN(backbone=backbone, encoded_image_size=14, fine_tune=False)
    encoder = encoder.to(device)
    encoder.eval()

    transform = get_val_transforms()
    dataset = ImageListDataset(images_to_process, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    saved_count = len(all_images) - len(images_to_process)
    with torch.no_grad():
        for batch_tensors, batch_names, valids in tqdm(dataloader, desc="Extracting features"):
            batch_tensors = batch_tensors.to(device)
            features = encoder(batch_tensors)  # (B, 196, feature_dim)

            features_cpu = features.cpu()
            for i, name in enumerate(batch_names):
                if valids[i]:
                    stem = Path(name).stem
                    out_path = features_dir / f"{stem}.pt"
                    # Crucial: clone().half() to save only the single slice (0.8 MB) instead of the full 50MB batch buffer
                    feat_slice = features_cpu[i].clone().half()
                    try:
                        torch.save(feat_slice, out_path, _use_new_zipfile_serialization=False)
                        saved_count += 1
                    except Exception as e:
                        import time
                        time.sleep(0.05)
                        try:
                            torch.save(feat_slice, out_path, _use_new_zipfile_serialization=False)
                            saved_count += 1
                        except Exception as e2:
                            logger.warning(f"Failed saving {out_path}: {e2}")

    logger.info(f"Successfully cached features for {saved_count} images in {features_dir}")
    return saved_count
