"""Spatial Attention Mechanism (Bahdanau Additive Attention)."""

import torch
import torch.nn as nn
from typing import Tuple


class BahdanauAttention(nn.Module):
    """Bahdanau Additive Spatial Attention Module.
    
    Computes dynamic attention alignment scores over image spatial regions given
    the current decoder hidden state.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates attention context and attention weights.
        
        Args:
            encoder_out: Tensor of shape (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Tensor of shape (batch_size, decoder_dim)
            
        Returns:
            attention_weighted_encoding: Tensor of shape (batch_size, encoder_dim)
            alpha: Attention weights of shape (batch_size, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)       # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)     # (batch_size, attention_dim)
        
        # Broadcast decoder_hidden over spatial locations
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        
        # Weighted sum across spatial locations
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        return attention_weighted_encoding, alpha
