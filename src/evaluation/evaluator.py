"""Batch evaluator for computing quantitative metrics across test datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inference.predictor import CaptionPredictor
from src.evaluation.metrics import evaluate_captions
from src.utils.logger import get_logger

logger = get_logger("evaluator")


def evaluate_dataset(
    predictor: CaptionPredictor,
    test_loader: DataLoader,
    output_json: Optional[Path] = None,
    method: str = "beam",
    beam_width: int = 5,
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """Runs batch evaluation over test dataset against 5 ground-truth references per image.
    
    Args:
        predictor: CaptionPredictor instance
        test_loader: DataLoader returning (img_tensor, references_list, img_name)
        output_json: Optional path to save evaluation results JSON
        method: 'beam' or 'greedy'
        beam_width: Beam width for beam search
        max_samples: Optional limit on test samples
        
    Returns:
        metrics_summary: Dictionary containing BLEU 1-4, ROUGE 1/2/L, and METEOR scores.
    """
    candidates: List[str] = []
    references_list: List[List[str]] = []
    sample_records: List[Dict] = []

    count = 0
    pbar = tqdm(test_loader, desc=f"Evaluating test set ({method})")
    for img_tensor, refs, img_name in pbar:
        flat_refs = [r[0] if isinstance(r, (list, tuple)) else str(r) for r in refs]

        result = predictor.predict(img_tensor, method=method, beam_width=beam_width)
        cand_caption = result.caption

        candidates.append(cand_caption)
        references_list.append(flat_refs)

        sample_records.append({
            "image_name": img_name[0] if isinstance(img_name, (list, tuple)) else str(img_name),
            "generated_caption": cand_caption,
            "reference_captions": flat_refs,
            "confidence_score": result.confidence_score
        })

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    logger.info(f"Computing BLEU, ROUGE, and METEOR metrics for {len(candidates)} test samples...")
    metrics = evaluate_captions(candidates, references_list)

    logger.info(f"Evaluation Results:")
    logger.info(f"  BLEU-1:  {metrics['bleu_1']*100:.2f}%")
    logger.info(f"  BLEU-2:  {metrics['bleu_2']*100:.2f}%")
    logger.info(f"  BLEU-3:  {metrics['bleu_3']*100:.2f}%")
    logger.info(f"  BLEU-4:  {metrics['bleu_4']*100:.2f}%")
    logger.info(f"  ROUGE-1: {metrics['rouge_1']*100:.2f}%")
    logger.info(f"  ROUGE-2: {metrics['rouge_2']*100:.2f}%")
    logger.info(f"  ROUGE-L: {metrics['rouge_l']*100:.2f}%")
    logger.info(f"  METEOR:  {metrics['meteor']*100:.2f}%")

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "summary_metrics": metrics,
            "num_samples": len(candidates),
            "method": method,
            "beam_width": beam_width,
            "samples": sample_records[:50]  # Store top 50 sample predictions for qualitative inspection
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved evaluation report to {output_json}")

    return metrics
