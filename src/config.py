"""Configuration management for Image Caption Generator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import os


@dataclass
class PathConfig:
    """Directory and file paths configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset" if (Path(__file__).resolve().parent.parent / "dataset").exists() else Path(__file__).resolve().parent.parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    features_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "features")
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "checkpoints")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")
    vocab_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed" / "vocab.json")
    captions_file: Path = field(default_factory=lambda: (
        Path(__file__).resolve().parent.parent / "dataset" / "captions.txt"
        if (Path(__file__).resolve().parent.parent / "dataset" / "captions.txt").exists()
        else Path(__file__).resolve().parent.parent / "data" / "raw" / "captions.txt"
    ))
    images_dir: Path = field(default_factory=lambda: (
        Path(__file__).resolve().parent.parent / "dataset" / "Images"
        if (Path(__file__).resolve().parent.parent / "dataset" / "Images").exists()
        else Path(__file__).resolve().parent.parent / "data" / "raw" / "Images"
    ))

    def __post_init__(self):
        for dir_path in [self.data_dir, self.processed_data_dir, 
                         self.features_dir, self.checkpoints_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing and tokenization parameters."""
    image_size: Tuple[int, int] = (224, 224)
    min_word_freq: int = 3
    max_caption_len: int = 35
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    pad_token: str = "<pad>"
    start_token: str = "<start>"
    end_token: str = "<end>"
    unk_token: str = "<unk>"
    batch_size: int = 32
    num_workers: int = 0
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Encoder, Decoder, and Attention architecture settings."""
    encoder_name: str = "resnet50"  # 'resnet50', 'mobilenet_v3_large', 'efficientnet_b0'
    encoder_dim: int = 2048         # 2048 for ResNet50, 960 for MobileNetV3-Large, 1280 for EfficientNet-B0
    embedding_dim: int = 512
    attention_dim: int = 512
    decoder_dim: int = 512
    dropout: float = 0.3
    vocab_size: int = 5000
    fine_tune_encoder: bool = False
    encoder_lr: float = 1e-4
    decoder_lr: float = 4e-4


@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""
    num_epochs: int = 15
    grad_clip: float = 5.0
    alpha_c: float = 1.0  # Doubly stochastic attention regularization weight
    patience: int = 5
    min_delta: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    save_top_k: int = 3
    seed: int = 42


@dataclass
class InferenceConfig:
    """Parameters for caption generation."""
    beam_width: int = 5
    max_len: int = 30
    temperature: float = 1.0
    length_penalty_alpha: float = 0.7
    repetition_penalty: float = 1.2


@dataclass
class AppConfig:
    """API and Streamlit deployment configuration."""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    model_checkpoint: Optional[str] = None


@dataclass
class GlobalConfig:
    """Aggregated global configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    app: AppConfig = field(default_factory=AppConfig)


def get_default_config() -> GlobalConfig:
    """Factory helper to obtain a default instantiated config."""
    return GlobalConfig()
