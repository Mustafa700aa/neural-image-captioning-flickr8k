"""Integration tests for Greedy Search, Beam Search, and CaptionPredictor."""

import torch
from PIL import Image
from src.data.vocabulary import Vocabulary
from src.models.captioner import ImageCaptionModel
from src.inference.predictor import CaptionPredictor
from src.inference.greedy_search import greedy_decode
from src.inference.beam_search import beam_search_decode


def test_greedy_and_beam_search_decoding():
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(["a dog is running in the grass", "a cat sleeps"])
    
    model = ImageCaptionModel(
        vocab_size=len(vocab),
        encoder_dim=512,
        decoder_dim=128,
        attention_dim=128,
        embed_dim=128
    )
    model.eval()

    encoder_out = torch.randn(1, 49, 512)

    # Test Greedy
    g_idx, g_words, g_alphas = greedy_decode(
        decoder=model.decoder,
        encoder_out=encoder_out,
        vocab=vocab,
        max_len=10
    )
    assert isinstance(g_words, list)
    assert g_alphas.dim() == 2

    # Test Beam
    b_idx, b_words, b_alphas, score = beam_search_decode(
        decoder=model.decoder,
        encoder_out=encoder_out,
        vocab=vocab,
        beam_width=3,
        max_len=10
    )
    assert isinstance(b_words, list)
    assert b_alphas.dim() == 2
    assert isinstance(score, float)


def test_caption_predictor_with_pil_image():
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(["a cute puppy playing with a toy"])

    model = ImageCaptionModel(
        vocab_size=len(vocab),
        encoder_dim=2048,
        decoder_dim=256,
        attention_dim=256,
        embed_dim=256
    )

    predictor = CaptionPredictor(model=model, vocab=vocab)

    dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    res_beam = predictor.predict(dummy_img, method="beam", beam_width=3)
    res_greedy = predictor.predict(dummy_img, method="greedy")

    assert isinstance(res_beam.caption, str)
    assert isinstance(res_greedy.caption, str)
    assert isinstance(res_beam.tokens, list)
    assert isinstance(res_greedy.tokens, list)
    assert res_beam.confidence_score is not None
