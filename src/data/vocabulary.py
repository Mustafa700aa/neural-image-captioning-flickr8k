"""Vocabulary construction, tokenization, numericalization, and serialization."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union


class Vocabulary:
    """Manages word-to-index and index-to-word mappings with special token handling."""

    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freqs: Counter = Counter()

        # Initialize special tokens
        self.pad_idx = self._add_word(self.PAD_TOKEN)
        self.start_idx = self._add_word(self.START_TOKEN)
        self.end_idx = self._add_word(self.END_TOKEN)
        self.unk_idx = self._add_word(self.UNK_TOKEN)

    def _add_word(self, word: str) -> int:
        """Adds a single word to the vocabulary if not present."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    @staticmethod
    def clean_caption(caption: str) -> str:
        """Cleans raw caption text: lowercasing, punctuation normalization, spacing."""
        caption = caption.lower().strip()
        # Replace punctuation with whitespace except keeping essential alphanumerics
        caption = re.sub(r"[^a-zA-Z0-9\s]", " ", caption)
        # Collapse multiple spaces
        caption = re.sub(r"\s+", " ", caption).strip()
        return caption

    @staticmethod
    def tokenize(caption: str) -> List[str]:
        """Tokenizes a cleaned caption string into a list of words."""
        cleaned = Vocabulary.clean_caption(caption)
        return cleaned.split() if cleaned else []

    def build_vocabulary(self, sentence_list: List[str]) -> "Vocabulary":
        """Builds vocabulary from a corpus of raw or cleaned caption sentences."""
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            self.word_freqs.update(tokens)

        # Add words meeting minimum frequency threshold
        for word, freq in self.word_freqs.items():
            if freq >= self.min_freq:
                self._add_word(word)

        return self

    def numericalize(self, caption: Union[str, List[str]], add_special_tokens: bool = True) -> List[int]:
        """Converts a caption string or token list to a list of token indices."""
        if isinstance(caption, str):
            tokens = self.tokenize(caption)
        else:
            tokens = caption

        indices = [self.word2idx.get(w, self.unk_idx) for w in tokens]

        if add_special_tokens:
            return [self.start_idx] + indices + [self.end_idx]
        return indices

    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """Converts token indices back into a readable sentence string."""
        words = []
        for idx in indices:
            # Handle PyTorch tensor or scalar int
            if hasattr(idx, "item"):
                idx = idx.item()
            word = self.idx2word.get(int(idx), self.UNK_TOKEN)
            if remove_special_tokens and word in (self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN):
                if word == self.END_TOKEN:
                    break
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves vocabulary structure, mappings, and word frequencies to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_freqs": dict(self.word_freqs),
            "pad_idx": self.pad_idx,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "unk_idx": self.unk_idx,
            "vocab_size": len(self.word2idx)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Loads vocabulary from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(min_freq=data.get("min_freq", 3))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.word_freqs = Counter(data.get("word_freqs", {}))
        vocab.pad_idx = data["pad_idx"]
        vocab.start_idx = data["start_idx"]
        vocab.end_idx = data["end_idx"]
        vocab.unk_idx = data["unk_idx"]
        return vocab
