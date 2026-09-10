"""End-to-end Image Captioning Model container."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.models.encoder import EncoderCNN
from src.models.decoder import DecoderWithAttention
from src.config import ModelConfig


class ImageCaptionModel(nn.Module):
    """End-to-End Image Captioning System connecting Vision Encoder with Attention Decoder."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        vocab_size: int = 5000,
        encoder_dim: int = 2048,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        embed_dim: int = 512,
        dropout: float = 0.3,
        backbone: str = "resnet50",
        fine_tune_encoder: bool = False
    ):
        super().__init__()
        
        if config is not None:
            self.vocab_size = config.vocab_size
            self.encoder_dim = config.encoder_dim
            self.decoder_dim = config.decoder_dim
            self.attention_dim = config.attention_dim
            self.embed_dim = config.embedding_dim
            self.dropout = config.dropout
            self.backbone = config.encoder_name
            self.fine_tune_enc = config.fine_tune_encoder
        else:
            self.vocab_size = vocab_size
            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim
            self.attention_dim = attention_dim
            self.embed_dim = embed_dim
            self.dropout = dropout
            self.backbone = backbone
            self.fine_tune_enc = fine_tune_encoder

        self.encoder = EncoderCNN(
            backbone=self.backbone,
            encoded_image_size=14,
            fine_tune=self.fine_tune_enc
        )

        self.decoder = DecoderWithAttention(
            attention_dim=self.attention_dim,
            embed_dim=self.embed_dim,
            decoder_dim=self.decoder_dim,
            vocab_size=self.vocab_size,
            encoder_dim=self.encoder_dim,
            dropout=self.dropout
        )

    def forward(
        self,
        images: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and decoder.
        
        Args:
            images: Tensor of shape (batch_size, 3, H, W) OR precomputed features (batch_size, num_pixels, encoder_dim)
            encoded_captions: Tensor of shape (batch_size, max_caption_len)
            caption_lengths: Tensor of shape (batch_size,)
        """
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, encoded_captions, caption_lengths)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts spatial visual features from images using the encoder."""
        return self.encoder(images)
