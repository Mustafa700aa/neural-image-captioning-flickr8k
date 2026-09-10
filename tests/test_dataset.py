"""Unit tests for Dataset loading, splitting, and batch collation."""

import pandas as pd
import torch
from src.data.dataset import split_flickr8k_data, CaptionCollate, Flickr8kDataset
from src.data.vocabulary import Vocabulary


def test_split_without_data_leakage():
    # 10 images with 5 captions each = 50 rows
    records = []
    for img_id in range(10):
        for cap_id in range(5):
            records.append({
                "image_name": f"image_{img_id:02d}.jpg",
                "caption": f"caption {cap_id} for image {img_id}"
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = split_flickr8k_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    train_images = set(train_df["image_name"])
    val_images = set(val_df["image_name"])
    test_images = set(test_df["image_name"])

    # Disjoint check - zero overlap between image sets
    assert len(train_images.intersection(val_images)) == 0
    assert len(train_images.intersection(test_images)) == 0
    assert len(val_images.intersection(test_images)) == 0
    assert len(train_images) == 8
    assert len(val_images) == 1
    assert len(test_images) == 1


def test_caption_collate_dynamic_padding():
    pad_idx = 0
    collate_fn = CaptionCollate(pad_idx=pad_idx)

    # 3 dummy samples with different sequence lengths
    item1 = (torch.zeros(3, 224, 224), torch.tensor([1, 10, 20, 2]), "img1.jpg")
    item2 = (torch.zeros(3, 224, 224), torch.tensor([1, 10, 20, 30, 40, 2]), "img2.jpg")
    item3 = (torch.zeros(3, 224, 224), torch.tensor([1, 10, 2]), "img3.jpg")

    batch = [item1, item2, item3]
    images, targets, lengths, img_names = collate_fn(batch)

    assert images.shape == (3, 3, 224, 224)
    # Longest sequence has length 6
    assert targets.shape == (3, 6)
    assert targets[0, 5] == 2  # item2 (len 6) ends at index 5
    assert targets[1, 3] == 2  # item1 (len 4) ends at index 3
    assert targets[2, 3] == pad_idx  # item3 (len 3) has padding at index 3, 4, 5
    assert lengths.tolist() == [6, 4, 3]  # sorted descending
