"""FastAPI Application for Image Captioning Model Deployment."""

import base64
import io
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (
    Base64PredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse
)
from src.config import get_default_config
from src.inference.predictor import CaptionPredictor
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger("api")
cfg = get_default_config()

app = FastAPI(
    title="Image Caption Generator API",
    description="Production REST API for generating captions from images using CNN + Bahdanau Spatial Attention + LSTM.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global Predictor instance
_predictor: Optional[CaptionPredictor] = None


def get_predictor() -> CaptionPredictor:
    """Lazy loader and singleton accessor for the model predictor."""
    global _predictor
    if _predictor is None:
        logger.info("Initializing CaptionPredictor for API...")
        best_ckpt = cfg.paths.checkpoints_dir / "caption_model_best.pt"
        ckpt_path = best_ckpt if best_ckpt.exists() else None

        _predictor = CaptionPredictor(
            checkpoint_path=ckpt_path,
            vocab_path=cfg.paths.vocab_path if cfg.paths.vocab_path.exists() else None,
            config=cfg,
            device=get_device()
        )
    return _predictor


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint returning system status and device info."""
    predictor = get_predictor()
    device = str(predictor.device)
    return HealthResponse(
        status="healthy",
        device=device,
        model_loaded=predictor.model is not None,
        vocab_size=len(predictor.vocab),
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info():
    """Returns architecture and model parameter configurations."""
    predictor = get_predictor()
    return ModelInfoResponse(
        model_name="Image Caption Generator (CNN-LSTM-Attention)",
        encoder_backbone=cfg.model.encoder_name,
        encoder_dim=cfg.model.encoder_dim,
        decoder_dim=cfg.model.decoder_dim,
        attention_dim=cfg.model.attention_dim,
        vocab_size=len(predictor.vocab),
        fine_tuned=cfg.model.fine_tune_encoder
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    method: str = Form(default="beam", description="Decoding method ('beam' or 'greedy')"),
    beam_width: int = Form(default=5, description="Beam search width"),
    max_len: int = Form(default=30, description="Max sequence length"),
    temperature: float = Form(default=1.0, description="Sampling temperature")
):
    """Generates a caption for an uploaded image file."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    start_time = time.time()
    predictor = get_predictor()

    try:
        result = predictor.predict(
            image=pil_img,
            method=method,
            beam_width=beam_width,
            max_len=max_len,
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000.0

    return PredictionResponse(
        caption=result.caption,
        tokens=result.tokens,
        token_indices=result.token_indices,
        confidence_score=result.confidence_score,
        method=result.method,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/predict-base64", response_model=PredictionResponse, tags=["Inference"])
def predict_base64(payload: Base64PredictionRequest):
    """Generates a caption from a Base64-encoded image string."""
    try:
        # Strip header if present
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

    start_time = time.time()
    predictor = get_predictor()

    result = predictor.predict(
        image=pil_img,
        method=payload.method,
        beam_width=payload.beam_width,
        max_len=payload.max_len,
        temperature=payload.temperature
    )

    latency_ms = (time.time() - start_time) * 1000.0

    return PredictionResponse(
        caption=result.caption,
        tokens=result.tokens,
        token_indices=result.token_indices,
        confidence_score=result.confidence_score,
        method=result.method,
        latency_ms=round(latency_ms, 2)
    )
