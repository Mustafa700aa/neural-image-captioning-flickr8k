"""Pydantic schemas for FastAPI service."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CaptionTokenInfo(BaseModel):
    token: str
    token_id: int


class PredictionResponse(BaseModel):
    caption: str = Field(..., description="Generated natural-language image caption")
    tokens: List[str] = Field(..., description="List of generated word tokens")
    token_indices: List[int] = Field(..., description="Numerical token IDs")
    confidence_score: float = Field(..., description="Sequence confidence / log probability score")
    method: str = Field(..., description="Decoding algorithm used (beam or greedy)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class Base64PredictionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string")
    method: str = Field(default="beam", description="Decoding method ('beam' or 'greedy')")
    beam_width: int = Field(default=5, ge=1, le=20, description="Beam search width")
    max_len: int = Field(default=30, ge=5, le=50, description="Maximum caption length")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature for greedy decoding")


class HealthResponse(BaseModel):
    status: str = "healthy"
    device: str
    model_loaded: bool
    vocab_size: int
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_name: str = "Image Caption Generator (CNN-LSTM-Attention)"
    encoder_backbone: str
    encoder_dim: int
    decoder_dim: int
    attention_dim: int
    vocab_size: int
    fine_tuned: bool
