"""High-level CaptionPredictor for inference pipelines and serving."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import io
from PIL import Image
import torch

from src.models.captioner import ImageCaptionModel
from src.data.vocabulary import Vocabulary
from src.data.transforms import get_inference_transforms
from src.inference.beam_search import beam_search_decode
from src.inference.greedy_search import greedy_decode
from src.config import GlobalConfig, get_default_config
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger("predictor")


@dataclass
class CaptionResult:
    """Structured result returned by CaptionPredictor."""
    caption: str
    tokens: List[str]
    token_indices: List[int]
    confidence_score: float
    attention_weights: torch.Tensor  # (seq_len, num_pixels)
    method: str


class CaptionPredictor:
    """Production inference interface for generating image captions with visual attention."""

    def __init__(
        self,
        model: Optional[ImageCaptionModel] = None,
        vocab: Optional[Vocabulary] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
        config: Optional[GlobalConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.cfg = config or get_default_config()
        self.device = device or get_device()
        self.transform = get_inference_transforms(image_size=self.cfg.data.image_size)

        # Load Vocabulary
        if vocab is not None:
            self.vocab = vocab
        elif vocab_path is not None and Path(vocab_path).exists():
            self.vocab = Vocabulary.load(vocab_path)
        elif self.cfg.paths.vocab_path.exists():
            self.vocab = Vocabulary.load(self.cfg.paths.vocab_path)
        else:
            self.vocab = Vocabulary()

        # Initialize Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = ImageCaptionModel(
                vocab_size=max(len(self.vocab), 100),
                encoder_dim=self.cfg.model.encoder_dim,
                decoder_dim=self.cfg.model.decoder_dim,
                attention_dim=self.cfg.model.attention_dim,
                embed_dim=self.cfg.model.embedding_dim,
                dropout=self.cfg.model.dropout,
                backbone=self.cfg.model.encoder_name,
                fine_tune_encoder=False
            ).to(self.device)

            if checkpoint_path is not None and Path(checkpoint_path).exists():
                logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def _prepare_image(self, image_input: Union[str, Path, Image.Image, bytes, torch.Tensor]) -> torch.Tensor:
        """Preprocesses various image formats into a normalized torch tensor of shape (1, 3, H, W) or (1, num_pixels, dim)."""
        if isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3 and image_input.size(0) == 3:
                return image_input.unsqueeze(0).to(self.device)
            elif image_input.dim() == 3:  # Already spatial feature (1, num_pixels, dim)
                return image_input.to(self.device)
            elif image_input.dim() == 4:
                return image_input.to(self.device)

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def predict(
        self,
        image: Union[str, Path, Image.Image, bytes, torch.Tensor],
        method: str = "beam",
        beam_width: int = 5,
        max_len: int = 30,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2
    ) -> CaptionResult:
        """Generates a caption for an input image using either Beam Search or Greedy Search.
        
        Args:
            image: Image path, PIL Image, bytes, or preprocessed Tensor
            method: 'beam' or 'greedy'
            beam_width: Number of beam paths (for beam search)
            max_len: Maximum token length
            temperature: Sampling temperature (for greedy search)
            repetition_penalty: Repetition penalty coefficient
            
        Returns:
            CaptionResult dataclass with caption, tokens, score, and attention maps.
        """
        img_tensor = self._prepare_image(image)

        with torch.no_grad():
            encoder_out = self.model.extract_features(img_tensor)  # (1, num_pixels, encoder_dim)

            if method.lower() == "greedy":
                indices, tokens, alphas = greedy_decode(
                    decoder=self.model.decoder,
                    encoder_out=encoder_out,
                    vocab=self.vocab,
                    max_len=max_len,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )
                score = 1.0
            else:
                indices, tokens, alphas, score = beam_search_decode(
                    decoder=self.model.decoder,
                    encoder_out=encoder_out,
                    vocab=self.vocab,
                    beam_width=beam_width,
                    max_len=max_len,
                    length_penalty_alpha=self.cfg.inference.length_penalty_alpha
                )

        caption = " ".join(tokens)
        return CaptionResult(
            caption=caption,
            tokens=tokens,
            token_indices=indices,
            confidence_score=float(score),
            attention_weights=alphas,
            method=method
        )
