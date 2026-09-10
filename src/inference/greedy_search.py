"""Greedy search autoregressive decoder for caption generation."""

from typing import List, Tuple
import torch
import torch.nn.functional as F

from src.models.decoder import DecoderWithAttention
from src.data.vocabulary import Vocabulary


def greedy_decode(
    decoder: DecoderWithAttention,
    encoder_out: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 30,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2
) -> Tuple[List[int], List[str], torch.Tensor]:
    """Generates caption using greedy autoregressive search.
    
    Args:
        decoder: Trained DecoderWithAttention module
        encoder_out: Feature tensor of shape (1, num_pixels, encoder_dim)
        vocab: Vocabulary instance
        max_len: Maximum tokens to generate
        temperature: Softmax sampling temperature (1.0 = standard argmax)
        repetition_penalty: Penalty multiplier for repeating recently generated tokens
        
    Returns:
        generated_indices: List of token IDs
        generated_words: List of word tokens
        alphas: Attention weights tensor of shape (seq_len, num_pixels)
    """
    decoder.eval()
    device = encoder_out.device

    # Initialize LSTM states
    h, c = decoder.init_hidden_state(encoder_out)

    # Start with <start> token
    current_token = torch.tensor([vocab.start_idx], dtype=torch.long, device=device)
    
    generated_indices: List[int] = []
    generated_words: List[str] = []
    alphas_list: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(max_len):
            embedding = decoder.embedding(current_token)  # (1, embed_dim)
            scores, h, c, alpha = decoder.step(embedding, encoder_out, h, c)  # scores: (1, vocab_size)

            # Apply repetition penalty to previously generated tokens
            if repetition_penalty != 1.0 and generated_indices:
                for prev_idx in set(generated_indices):
                    if scores[0, prev_idx] > 0:
                        scores[0, prev_idx] /= repetition_penalty
                    else:
                        scores[0, prev_idx] *= repetition_penalty

            if temperature != 1.0 and temperature > 0:
                scores = scores / temperature

            next_token = torch.argmax(scores, dim=1).item()

            alphas_list.append(alpha.squeeze(0).cpu())

            if next_token == vocab.end_idx:
                break

            generated_indices.append(next_token)
            word = vocab.idx2word.get(next_token, vocab.UNK_TOKEN)
            generated_words.append(word)

            current_token = torch.tensor([next_token], dtype=torch.long, device=device)

    alphas_tensor = torch.stack(alphas_list, dim=0) if alphas_list else torch.empty((0, encoder_out.size(1)))
    return generated_indices, generated_words, alphas_tensor
