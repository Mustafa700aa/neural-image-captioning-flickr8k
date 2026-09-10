"""Beam search decoder for caption generation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.models.decoder import DecoderWithAttention
from src.data.vocabulary import Vocabulary


@dataclass
class BeamHypothesis:
    """Represents a candidate beam hypothesis during decoding."""
    tokens: List[int] = field(default_factory=list)
    score: float = 0.0
    h: Optional[torch.Tensor] = None
    c: Optional[torch.Tensor] = None
    alphas: List[torch.Tensor] = field(default_factory=list)
    completed: bool = False


def beam_search_decode(
    decoder: DecoderWithAttention,
    encoder_out: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 5,
    max_len: int = 30,
    length_penalty_alpha: float = 0.7
) -> Tuple[List[int], List[str], torch.Tensor, float]:
    """Generates caption using Beam Search with length penalty normalization.
    
    Args:
        decoder: Trained DecoderWithAttention module
        encoder_out: Feature tensor of shape (1, num_pixels, encoder_dim)
        vocab: Vocabulary instance
        beam_width: Number of beam hypotheses to maintain (k)
        max_len: Maximum generation sequence length
        length_penalty_alpha: Exponential length penalty coefficient (0.0 = no penalty, 1.0 = linear)
        
    Returns:
        best_indices: Token IDs of best hypothesis
        best_words: Word tokens of best hypothesis
        best_alphas: Attention maps of shape (seq_len, num_pixels)
        best_score: Normalized sequence log probability score
    """
    decoder.eval()
    device = encoder_out.device

    # Initial LSTM states
    h0, c0 = decoder.init_hidden_state(encoder_out)

    # Initial hypothesis with <start> token
    beams: List[BeamHypothesis] = [
        BeamHypothesis(
            tokens=[vocab.start_idx],
            score=0.0,
            h=h0,
            c=c0,
            alphas=[]
        )
    ]

    completed_hypotheses: List[BeamHypothesis] = []

    with torch.no_grad():
        for step in range(max_len):
            all_candidates: List[BeamHypothesis] = []

            for hyp in beams:
                if hyp.completed:
                    all_candidates.append(hyp)
                    continue

                last_token = hyp.tokens[-1]
                token_tensor = torch.tensor([last_token], dtype=torch.long, device=device)
                embedding = decoder.embedding(token_tensor)  # (1, embed_dim)

                scores, next_h, next_c, alpha = decoder.step(
                    embedding,
                    encoder_out,
                    hyp.h,
                    hyp.c
                )
                log_probs = F.log_softmax(scores, dim=1).squeeze(0)  # (vocab_size,)

                # Top-k candidate tokens for this beam
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for k in range(beam_width):
                    token_idx = topk_indices[k].item()
                    prob = topk_log_probs[k].item()

                    new_tokens = hyp.tokens + [token_idx]
                    new_score = hyp.score + prob
                    new_alphas = hyp.alphas + [alpha.squeeze(0).cpu()]
                    is_completed = (token_idx == vocab.end_idx)

                    candidate = BeamHypothesis(
                        tokens=new_tokens,
                        score=new_score,
                        h=next_h,
                        c=next_c,
                        alphas=new_alphas,
                        completed=is_completed
                    )

                    if is_completed:
                        completed_hypotheses.append(candidate)
                    else:
                        all_candidates.append(candidate)

            # Prune and keep top-k beams
            def get_norm_score(h_cand: BeamHypothesis) -> float:
                length = len(h_cand.tokens) - 1  # exclude start token
                lp = ((5.0 + max(1, length)) / 6.0) ** length_penalty_alpha
                return h_cand.score / lp

            # Sort candidate beams by normalized score
            all_candidates.sort(key=get_norm_score, reverse=True)
            beams = all_candidates[:beam_width]

            if not beams or all(b.completed for b in beams):
                break

    # If no hypotheses finished with <end>, use top active beam
    all_final = completed_hypotheses if completed_hypotheses else beams
    all_final.sort(key=lambda h: h.score / (((5.0 + max(1, len(h.tokens) - 1)) / 6.0) ** length_penalty_alpha), reverse=True)
    best_hyp = all_final[0]

    # Filter out <start> and <end> tokens
    clean_indices = [idx for idx in best_hyp.tokens if idx not in (vocab.start_idx, vocab.end_idx)]
    clean_words = [vocab.idx2word.get(idx, vocab.UNK_TOKEN) for idx in clean_indices]
    
    alphas_tensor = torch.stack(best_hyp.alphas, dim=0) if best_hyp.alphas else torch.empty((0, encoder_out.size(1)))
    # Slice alphas to match clean words length
    if alphas_tensor.size(0) > len(clean_words):
        alphas_tensor = alphas_tensor[:len(clean_words)]

    final_score = best_hyp.score / (((5.0 + max(1, len(clean_indices))) / 6.0) ** length_penalty_alpha)

    return clean_indices, clean_words, alphas_tensor, final_score
