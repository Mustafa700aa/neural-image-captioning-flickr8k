"""HuggingFace Hub integration for model storage, sharing, and checkpoint loading."""

import json
from pathlib import Path
from typing import Optional, Union
import torch

from src.config import GlobalConfig, get_default_config
from src.utils.logger import get_logger

logger = get_logger("hf_hub")


def push_model_to_hub(
    repo_id: str,
    checkpoint_path: Union[str, Path],
    vocab_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
    private: bool = False
) -> str:
    """Uploads trained Image Captioning checkpoint, vocabulary, and model card to HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository identifier (e.g. 'username/image-caption-flickr8k')
        checkpoint_path: Path to caption_model_best.pt
        vocab_path: Path to vocab.json
        config_path: Path to config JSON or dataclass dump
        metrics_path: Path to evaluation_results.json
        token: HuggingFace authentication token
        private: Whether the HuggingFace repo should be private
        
    Returns:
        repo_url: URL to the public/shareable repository
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
    except ImportError:
        logger.error("huggingface_hub library is not installed. Install with: pip install huggingface_hub")
        raise

    api = HfApi(token=token)
    
    logger.info(f"Creating/verifying HuggingFace repository: {repo_id}...")
    repo_url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

    # 1. Upload Model Checkpoint
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        logger.info(f"Uploading model checkpoint ({checkpoint_path.name})...")
        upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="caption_model_best.pt",
            repo_id=repo_id,
            token=token
        )

    # 2. Upload Vocabulary
    vocab_path = Path(vocab_path)
    if vocab_path.exists():
        logger.info("Uploading vocabulary (vocab.json)...")
        upload_file(
            path_or_fileobj=str(vocab_path),
            path_in_repo="vocab.json",
            repo_id=repo_id,
            token=token
        )

    # 3. Upload Metrics & Model Card
    readme_content = f"""---
language:
- en
license: apache-2.0
tags:
- image-captioning
- computer-vision
- nlp
- pytorch
- resnet50
- attention
- flickr8k
datasets:
- flickr8k
metrics:
- bleu
- rouge
- meteor
pipeline_tag: image-to-text
---

# Neural Image Caption Generator (Flickr8k)

This repository contains the trained weights and vocabulary for the **CNN-Attention-LSTM Image Caption Generator** trained on the Flickr8k dataset.

## Model Architecture
- **Vision Backbone**: Pretrained ResNet-50 (Transfer Learning)
- **Spatial Grid**: $14 \\times 14 \\times 2048$ (196 spatial locations)
- **Attention Mechanism**: Bahdanau Additive Attention with Adaptive Gating
- **Decoder**: LSTM with Word Embeddings (512-dim)
- **Regularization**: Doubly Stochastic Attention Regularization

## Quick Usage

```python
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

# Download artifacts
ckpt_path = hf_hub_download(repo_id="{repo_id}", filename="caption_model_best.pt")
vocab_path = hf_hub_download(repo_id="{repo_id}", filename="vocab.json")

# Load with src.inference.predictor
from src.inference.predictor import CaptionPredictor
predictor = CaptionPredictor(checkpoint_path=ckpt_path, vocab_path=vocab_path)

image = Image.open("sample.jpg").convert("RGB")
result = predictor.predict(image, method="beam", beam_width=5)
print("Generated Caption:", result.caption)
```
"""
    logger.info("Uploading Model Card (README.md)...")
    upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token
    )

    public_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model successfully published to HuggingFace Hub: {public_url}")
    return public_url
