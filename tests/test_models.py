"""Unit tests for CNN Encoder, Attention Mechanism, Decoder, and Loss Function."""

import torch
import pytest
from src.models.encoder import EncoderCNN
from src.models.attention import BahdanauAttention
from src.models.decoder import DecoderWithAttention
from src.models.captioner import ImageCaptionModel
from src.models.loss import CaptionLoss


def test_encoder_spatial_feature_output():
    encoder = EncoderCNN(backbone="resnet50", encoded_image_size=14, fine_tune=False)
    encoder.eval()

    dummy_images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = encoder(dummy_images)

    # Output shape should be (batch_size, 196, 2048)
    assert features.shape == (2, 196, 2048)


def test_bahdanau_attention_weights():
    encoder_dim = 2048
    decoder_dim = 512
    attention_dim = 512
    num_pixels = 196

    attention = BahdanauAttention(encoder_dim=encoder_dim, decoder_dim=decoder_dim, attention_dim=attention_dim)

    encoder_out = torch.randn(4, num_pixels, encoder_dim)
    decoder_hidden = torch.randn(4, decoder_dim)

    context, alpha = attention(encoder_out, decoder_hidden)

    assert context.shape == (4, encoder_dim)
    assert alpha.shape == (4, num_pixels)

    # Attention weights over spatial pixels must sum to 1.0 (softmax property)
    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_decoder_with_attention_forward():
    vocab_size = 100
    embed_dim = 256
    decoder_dim = 512
    encoder_dim = 2048
    attention_dim = 256
    batch_size = 3

    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim
    )

    encoder_out = torch.randn(batch_size, 196, encoder_dim)
    # 3 captions of length 6, 5, 4 (sorted descending)
    captions = torch.tensor([
        [1, 10, 20, 30, 40, 2],
        [1, 15, 25, 35, 2, 0],
        [1, 12, 22, 2, 0, 0]
    ], dtype=torch.long)
    lengths = torch.tensor([6, 5, 4], dtype=torch.long)

    scores, caps, decode_lengths, alphas = decoder(encoder_out, captions, lengths)

    max_decode_len = max(decode_lengths)
    assert max_decode_len == 5  # lengths - 1
    assert scores.shape == (batch_size, max_decode_len, vocab_size)
    assert alphas.shape == (batch_size, max_decode_len, 196)


def test_caption_loss_computation():
    pad_idx = 0
    loss_fn = CaptionLoss(pad_idx=pad_idx, alpha_c=1.0)

    batch_size = 2
    max_decode_len = 4
    vocab_size = 50

    scores = torch.randn(batch_size, max_decode_len, vocab_size, requires_grad=True)
    targets = torch.tensor([
        [1, 10, 20, 30, 2],
        [1, 15, 25, 2, 0]
    ], dtype=torch.long)
    decode_lengths = [4, 3]
    alphas = torch.softmax(torch.randn(batch_size, max_decode_len, 196), dim=2)

    total_loss, ce_loss, reg_loss = loss_fn(scores, targets, decode_lengths, alphas)

    assert total_loss.item() > 0
    assert ce_loss.item() > 0
    assert reg_loss.item() >= 0
    assert total_loss.requires_grad
