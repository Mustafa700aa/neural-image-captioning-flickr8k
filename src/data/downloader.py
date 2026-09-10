"""Flickr8k dataset downloader and synthetic dataset generator for local testing."""

import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("downloader")


def download_flickr8k(dest_dir: Path) -> Path:
    """Attempts to download Flickr8k using kagglehub or direct mirror."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import kagglehub
        logger.info("Attempting to download Flickr8k dataset via kagglehub...")
        path = kagglehub.dataset_download("adityajn105/flickr8k")
        logger.info(f"Flickr8k dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        logger.warning(f"Could not download via kagglehub: {e}. Checking local cache or generating sample data.")
        return dest_dir


def create_mock_flickr8k_dataset(
    output_dir: Path,
    num_samples: int = 50,
    image_size: tuple = (224, 224)
) -> Tuple[Path, Path]:
    """Generates a synthetic Flickr8k-like dataset with sample images and 5 captions per image.
    
    Useful for local testing, CI/CD validation, and initial development.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_file = output_dir / "captions.txt"

    sample_templates = [
        ("a brown dog is running through the green grass",
         "a cute dog playing on a lawn outside",
         "a brown dog chasing something in the yard",
         "a pet dog running in the park",
         "a furry dog outdoors on the grass"),
        ("a person riding a bicycle on a city street",
         "a cyclist wearing a helmet riding down the road",
         "someone commuting on a bike during the day",
         "a person on a bike next to traffic",
         "a bicycle rider navigating city street"),
        ("a group of children playing soccer on the field",
         "kids kicking a ball during a soccer match",
         "young boys and girls playing sports outdoors",
         "a children soccer team on the green pitch",
         "kids having fun playing football outside"),
        ("a woman sitting at an outdoor cafe with a cup of coffee",
         "a lady enjoying coffee outside a restaurant",
         "a person relaxing at a sidewalk cafe table",
         "a woman with a drink seated in the sun",
         "a cafe patron sitting outside with coffee"),
        ("a yellow kayak paddling through clear blue water",
         "a small boat on a calm mountain lake",
         "a kayaker paddling in sunny weather",
         "a bright yellow boat on the water surface",
         "an outdoor adventurer kayaking on the lake")
    ]

    colors = [
        (76, 175, 80),   # Green
        (33, 150, 243),  # Blue
        (255, 152, 0),   # Orange
        (233, 30, 99),   # Pink
        (156, 39, 176),  # Purple
        (0, 188, 212),   # Cyan
        (255, 87, 34)    # Deep Orange
    ]

    records = []
    for i in range(num_samples):
        img_name = f"sample_{i:04d}.jpg"
        img_path = images_dir / img_name

        # Create synthetic image with gradient and shapes
        bg_color = colors[i % len(colors)]
        img = Image.new("RGB", image_size, color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add decorative shapes
        draw.ellipse([20, 20, 80, 80], fill=(255, 255, 255, 180), outline=(0, 0, 0))
        draw.rectangle([100, 120, 200, 180], fill=(240, 240, 240), outline=(50, 50, 50))
        draw.text((30, 190), f"Sample #{i+1}", fill=(255, 255, 255))
        img.save(img_path, quality=90)

        # Assign 5 reference captions
        template = sample_templates[i % len(sample_templates)]
        for cap in template:
            records.append({"image": img_name, "caption": cap})

    df = pd.DataFrame(records)
    df.to_csv(captions_file, index=False)
    logger.info(f"Created synthetic dataset at {output_dir} with {num_samples} images and {len(records)} captions.")

    return images_dir, captions_file
