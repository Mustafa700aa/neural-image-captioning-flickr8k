"""Unit tests for BLEU, ROUGE, and METEOR evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    compute_sentence_bleu,
    compute_corpus_bleu,
    compute_rouge_scores,
    compute_meteor_scores,
    evaluate_captions
)


def test_sentence_bleu_exact_match():
    cand = "a brown dog running in the park"
    refs = [
        "a brown dog running in the park",
        "a cute dog playing outside in the park"
    ]
    score = compute_sentence_bleu(cand, refs)
    assert score > 0.9  # Near 1.0 for exact reference match


def test_corpus_bleu_scores():
    cands = [
        "a brown dog running across the lawn",
        "a little girl playing with a red ball"
    ]
    refs = [
        [
            "a brown dog is running through the green grass",
            "a cute dog running on a lawn outside",
            "a brown dog chasing a ball in the yard"
        ],
        [
            "a young girl playing with a red ball outside",
            "a small child holding a red ball",
            "a happy little girl tossing a ball"
        ]
    ]

    metrics = compute_corpus_bleu(cands, refs)
    assert "bleu_1" in metrics
    assert "bleu_2" in metrics
    assert "bleu_3" in metrics
    assert "bleu_4" in metrics
    assert metrics["bleu_1"] > 0.0
    assert metrics["bleu_1"] >= metrics["bleu_4"]  # Lower n-gram precision is typically higher


def test_rouge_and_meteor_scores():
    cands = ["a black cat sleeping on a sofa"]
    refs = [["a black cat is sleeping soundly on the couch", "a dark cat resting on the sofa"]]

    rouge = compute_rouge_scores(cands, refs)
    assert "rouge_1" in rouge
    assert "rouge_2" in rouge
    assert "rouge_l" in rouge
    assert rouge["rouge_1"] > 0.0

    meteor = compute_meteor_scores(cands, refs)
    assert meteor > 0.0


def test_evaluate_captions_bundle():
    cands = ["a cyclist riding a bicycle down the street"]
    refs = [["a cyclist on a bike down the road", "someone riding a bicycle in the city"]]

    bundle = evaluate_captions(cands, refs)
    assert "bleu_1" in bundle
    assert "rouge_l" in bundle
    assert "meteor" in bundle
