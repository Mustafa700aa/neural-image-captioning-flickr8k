"""Evaluation metrics for Image Captioning: BLEU (1-4), ROUGE (1, 2, L), and METEOR."""

from typing import Dict, List, Union
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Ensure nltk data is available if needed
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    try:
        nltk.download("wordnet", quiet=True)
    except Exception:
        pass


def compute_sentence_bleu(
    candidate: str,
    references: List[str],
    weights: tuple = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """Computes BLEU score for a single candidate caption against multiple references."""
    smooth = SmoothingFunction().method4
    cand_tokens = candidate.lower().split()
    ref_tokens = [ref.lower().split() for ref in references]
    
    if not cand_tokens:
        return 0.0
        
    return sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=smooth)


def compute_corpus_bleu(
    candidates: List[str],
    references_list: List[List[str]]
) -> Dict[str, float]:
    """Computes corpus-level BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores."""
    smooth = SmoothingFunction().method1
    
    tokenized_cands = [c.lower().split() for c in candidates]
    tokenized_refs = [[r.lower().split() for r in refs] for refs in references_list]

    b1 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    b2 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    b3 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth)
    b4 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "bleu_1": float(b1),
        "bleu_2": float(b2),
        "bleu_3": float(b3),
        "bleu_4": float(b4)
    }


def compute_rouge_scores(
    candidates: List[str],
    references_list: List[List[str]]
) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores across multiple references."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    r1_scores, r2_scores, rl_scores = [], [], []

    for cand, refs in zip(candidates, references_list):
        if not cand.strip():
            r1_scores.append(0.0)
            r2_scores.append(0.0)
            rl_scores.append(0.0)
            continue

        best_r1, best_r2, best_rl = 0.0, 0.0, 0.0
        for ref in refs:
            scores = scorer.score(ref, cand)
            best_r1 = max(best_r1, scores["rouge1"].fmeasure)
            best_r2 = max(best_r2, scores["rouge2"].fmeasure)
            best_rl = max(best_rl, scores["rougeL"].fmeasure)

        r1_scores.append(best_r1)
        r2_scores.append(best_r2)
        rl_scores.append(best_rl)

    return {
        "rouge_1": float(sum(r1_scores) / max(1, len(r1_scores))),
        "rouge_2": float(sum(r2_scores) / max(1, len(r2_scores))),
        "rouge_l": float(sum(rl_scores) / max(1, len(rl_scores)))
    }


def compute_meteor_scores(
    candidates: List[str],
    references_list: List[List[str]]
) -> float:
    """Computes average METEOR score across candidates and multi-references."""
    scores = []
    for cand, refs in zip(candidates, references_list):
        cand_tokens = cand.lower().split()
        if not cand_tokens:
            scores.append(0.0)
            continue

        best_m = 0.0
        for ref in refs:
            ref_tokens = ref.lower().split()
            try:
                m = nltk.translate.meteor_score.meteor_score([ref_tokens], cand_tokens)
                best_m = max(best_m, m)
            except Exception:
                # Fallback unigram overlap estimation if wordnet fails
                overlap = len(set(cand_tokens) & set(ref_tokens)) / max(1, len(set(ref_tokens)))
                best_m = max(best_m, overlap)
        scores.append(best_m)

    return float(sum(scores) / max(1, len(scores)))


def evaluate_captions(
    candidates: List[str],
    references_list: List[List[str]]
) -> Dict[str, float]:
    """Computes all standard captioning metrics: BLEU 1-4, ROUGE 1/2/L, and METEOR."""
    bleu_metrics = compute_corpus_bleu(candidates, references_list)
    rouge_metrics = compute_rouge_scores(candidates, references_list)
    meteor = compute_meteor_scores(candidates, references_list)

    return {
        **bleu_metrics,
        **rouge_metrics,
        "meteor": meteor
    }
