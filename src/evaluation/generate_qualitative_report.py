"""Generates qualitative evaluation reports and visual samples showing Input Image -> Generated Caption -> Reference Captions."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import List, Optional
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import get_default_config
from src.data.dataset import create_dataloaders, parse_flickr8k_captions
from src.data.vocabulary import Vocabulary
from src.inference.predictor import CaptionPredictor
from src.evaluation.metrics import evaluate_captions, compute_sentence_bleu
from src.evaluation.visualization import plot_attention_heatmaps
from src.utils.logger import get_logger

logger = get_logger("qualitative_report")


def generate_qualitative_examples(
    predictor: Optional[CaptionPredictor] = None,
    num_examples: int = 5,
    output_dir: Optional[Path] = None
) -> List[dict]:
    """Generates qualitative image-caption comparison samples with visualizations and metrics."""
    cfg = get_default_config()
    out_dir = output_dir or (cfg.paths.outputs_dir / "qualitative_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    if predictor is None:
        best_ckpt = cfg.paths.checkpoints_dir / "caption_model_best.pt"
        ckpt_path = best_ckpt if best_ckpt.exists() else None
        vocab_path = cfg.paths.vocab_path if cfg.paths.vocab_path.exists() else None
        predictor = CaptionPredictor(checkpoint_path=ckpt_path, vocab_path=vocab_path, config=cfg)

    # Load test dataset
    df = parse_flickr8k_captions(cfg.paths.captions_file)
    train_loader, val_loader, test_loader, vocab = create_dataloaders(cfg, vocab=predictor.vocab, df=df)

    results = []
    count = 0

    for img_tensor, refs, img_name in test_loader:
        if count >= num_examples:
            break

        name_str = img_name[0] if isinstance(img_name, (list, tuple)) else str(img_name)
        ref_list = [r[0] if isinstance(r, (list, tuple)) else str(r) for r in refs]

        img_path = cfg.paths.images_dir / name_str
        if not img_path.exists():
            continue

        raw_img = Image.open(img_path).convert("RGB")

        # Predict with Beam Search
        beam_res = predictor.predict(raw_img, method="beam", beam_width=5)
        # Predict with Greedy Search
        greedy_res = predictor.predict(raw_img, method="greedy")

        # Compute sample-level metrics against the 5 human reference captions
        metrics = evaluate_captions([beam_res.caption], [ref_list])

        # Generate Attention Heatmap Visualization
        fig_path = out_dir / f"sample_{count+1:02d}_{Path(name_str).stem}_attention.png"
        if beam_res.tokens and beam_res.attention_weights.size(0) > 0:
            plot_attention_heatmaps(
                image=raw_img,
                words=beam_res.tokens,
                alphas=beam_res.attention_weights,
                output_path=fig_path,
                smooth=True
            )

        # Generate side-by-side presentation card
        card_fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.3]})
        ax_img.imshow(raw_img)
        ax_img.axis("off")
        ax_img.set_title(f"Input: {name_str}", fontsize=11, fontweight="bold")

        txt_content = (
            f"GENERATED CAPTION (Beam Search, k=5):\n"
            f"\"{beam_res.caption}\"\n\n"
            f"GENERATED CAPTION (Greedy Search):\n"
            f"\"{greedy_res.caption}\"\n\n"
            f"HUMAN REFERENCE CAPTIONS (5 paired):\n"
        )
        for i, ref in enumerate(ref_list[:5], 1):
            txt_content += f"{i}. {ref}\n"

        txt_content += (
            f"\nMETRICS:\n"
            f"• BLEU-1: {metrics['bleu_1']*100:.1f}%  |  BLEU-4: {metrics['bleu_4']*100:.1f}%\n"
            f"• ROUGE-L: {metrics['rouge_l']*100:.1f}%  |  METEOR: {metrics['meteor']*100:.1f}%"
        )

        ax_txt.text(0.02, 0.95, txt_content, fontsize=10, verticalalignment="top", fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"))
        ax_txt.axis("off")
        
        card_path = out_dir / f"sample_{count+1:02d}_{Path(name_str).stem}_card.png"
        card_fig.tight_layout()
        card_fig.savefig(card_path, dpi=180, bbox_inches="tight")
        plt.close(card_fig)

        record = {
            "sample_index": count + 1,
            "image_name": name_str,
            "generated_caption_beam": beam_res.caption,
            "generated_caption_greedy": greedy_res.caption,
            "reference_captions": ref_list,
            "metrics": metrics,
            "card_image": str(card_path),
            "attention_image": str(fig_path) if fig_path.exists() else None
        }
        results.append(record)
        count += 1

    # Save summary report JSON
    report_file = out_dir / "qualitative_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Generated {len(results)} qualitative evaluation samples in {out_dir}")
    return results


if __name__ == "__main__":
    generate_qualitative_examples(num_examples=5)
