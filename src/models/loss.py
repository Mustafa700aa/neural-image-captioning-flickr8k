"""Loss functions for Image Captioning with Doubly Stochastic Regularization."""

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CaptionLoss(nn.Module):
    """Loss module calculating Cross-Entropy Loss with Doubly Stochastic Attention Regularization."""

    def __init__(self, pad_idx: int, alpha_c: float = 1.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.alpha_c = alpha_c
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        decode_lengths: List[int],
        alphas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates total loss, cross-entropy loss, and attention regularization penalty.
        
        Args:
            scores: Predicted logits of shape (batch_size, max_decode_len, vocab_size)
            targets: Target encoded captions of shape (batch_size, max_caption_len)
            decode_lengths: List of active decoding lengths per sample
            alphas: Attention weights of shape (batch_size, max_decode_len, num_pixels)
            
        Returns:
            total_loss: scalar tensor
            ce_loss: scalar cross-entropy loss tensor
            att_reg_loss: scalar doubly-stochastic attention loss tensor
        """
        # Since we decode from word 1 to end (excluding <start> token at index 0)
        # targets need to be shifted by 1 to match predictions
        targets_for_loss = targets[:, 1:]  # (batch_size, max_caption_len - 1)

        # Pack padded sequences to ignore padded steps efficiently
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
        targets_packed = pack_padded_sequence(targets_for_loss, decode_lengths, batch_first=True, enforce_sorted=False).data

        ce_loss = self.criterion(scores_packed, targets_packed)

        # Doubly stochastic attention penalty
        # Forces the model to pay attention to all regions of the image across time
        # alphas sum across time: (batch_size, num_pixels)
        att_reg_loss = self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        total_loss = ce_loss + att_reg_loss
        return total_loss, ce_loss, att_reg_loss
