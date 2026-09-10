"""Attention-based LSTM Decoder for Caption Generation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.models.attention import BahdanauAttention


class DecoderWithAttention(nn.Module):
    """Decoder RNN with Bahdanau Attention and Gating Mechanism."""

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.3
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # LSTMCell decoding step: input is concatenation of [embedding, context_vector]
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        # Linear layers to find initial states of LSTMCell from mean encoder spatial features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Gating mechanism (creates scalar weight between 0 and 1 for context vector)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Final classification projection layer
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates the initial hidden and cell states for the decoder LSTM from mean image features.
        
        Args:
            encoder_out: Tensor of shape (batch_size, num_pixels, encoder_dim)
            
        Returns:
            h0, c0: Tensors of shape (batch_size, decoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder_out))
        c = torch.tanh(self.init_c(mean_encoder_out))
        return h, c

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            encoder_out: Tensor of shape (batch_size, num_pixels, encoder_dim)
            encoded_captions: Tensor of shape (batch_size, max_caption_len)
            caption_lengths: Tensor of shape (batch_size,) containing true lengths
            
        Returns:
            predictions: Tensor of shape (batch_size, max_len - 1, vocab_size)
            encoded_captions: Sorted captions
            decode_lengths: List of lengths to decode
            alphas: Attention weights tensor of shape (batch_size, max_len - 1, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Embed all words in captions
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_len, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, so decode lengths are length - 1
        decode_lengths = (caption_lengths - 1).tolist()

        max_decode_len = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, num_pixels, device=encoder_out.device)

        # Step through sequence
        for t in range(max_decode_len):
            # Batch size at current step (active sequences)
            batch_size_t = sum([l > t for l in decode_lengths])
            
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # Gating scalar
            gated_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], gated_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas

    def step(
        self,
        embeddings_t: torch.Tensor,
        encoder_out: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step for autoregressive inference (greedy / beam search).
        
        Args:
            embeddings_t: Tensor of shape (batch_size, embed_dim)
            encoder_out: Tensor of shape (batch_size, num_pixels, encoder_dim)
            h, c: Tensors of shape (batch_size, decoder_dim)
            
        Returns:
            preds: (batch_size, vocab_size)
            h, c: Updated hidden and cell states
            alpha: Attention weights of shape (batch_size, num_pixels)
        """
        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))
        gated_encoding = gate * attention_weighted_encoding

        h, c = self.decode_step(
            torch.cat([embeddings_t, gated_encoding], dim=1),
            (h, c)
        )
        preds = self.fc(self.dropout(h))
        return preds, h, c, alpha
