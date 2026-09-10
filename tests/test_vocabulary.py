"""Unit tests for Vocabulary tokenization, numericalization, and serialization."""

import pytest
from src.data.vocabulary import Vocabulary


def test_special_tokens_initialization():
    vocab = Vocabulary(min_freq=1)
    assert vocab.pad_idx == 0
    assert vocab.start_idx == 1
    assert vocab.end_idx == 2
    assert vocab.unk_idx == 3
    assert len(vocab) == 4


def test_caption_cleaning_and_tokenization():
    raw_caption = "A Dog, running fast in the Park! 123."
    cleaned = Vocabulary.clean_caption(raw_caption)
    assert cleaned == "a dog running fast in the park 123"
    
    tokens = Vocabulary.tokenize(raw_caption)
    assert tokens == ["a", "dog", "running", "fast", "in", "the", "park", "123"]


def test_vocabulary_frequency_filtering():
    corpus = [
        "a dog is running",
        "a cat is sleeping",
        "a dog is barking"
    ]
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocabulary(corpus)

    # 'a' (3), 'is' (3), 'dog' (2) should be present
    # 'cat', 'sleeping', 'running', 'barking' have freq 1 and should not be added
    assert "a" in vocab.word2idx
    assert "is" in vocab.word2idx
    assert "dog" in vocab.word2idx
    assert "cat" not in vocab.word2idx
    assert "sleeping" not in vocab.word2idx


def test_numericalization_and_decoding():
    corpus = ["a playful puppy is playing on grass"]
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(corpus)

    sentence = "a playful puppy"
    indices = vocab.numericalize(sentence, add_special_tokens=True)
    
    assert indices[0] == vocab.start_idx
    assert indices[-1] == vocab.end_idx
    
    decoded = vocab.decode(indices, remove_special_tokens=True)
    assert decoded == "a playful puppy"


def test_unknown_token_handling():
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocabulary(["cat dog bird", "cat dog"])

    # 'elephant' is out of vocabulary
    indices = vocab.numericalize("elephant", add_special_tokens=False)
    assert indices == [vocab.unk_idx]


def test_vocabulary_serialization(tmp_path):
    corpus = ["the quick brown fox jumps over the lazy dog"]
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(corpus)

    save_file = tmp_path / "vocab.json"
    vocab.save(save_file)
    assert save_file.exists()

    loaded_vocab = Vocabulary.load(save_file)
    assert len(loaded_vocab) == len(vocab)
    assert loaded_vocab.word2idx == vocab.word2idx
    assert loaded_vocab.pad_idx == vocab.pad_idx
