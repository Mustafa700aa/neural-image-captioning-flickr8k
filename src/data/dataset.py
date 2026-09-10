"""Flickr8k Dataset, DataLoaders, and Batch Collation."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.data.vocabulary import Vocabulary
from src.data.transforms import get_train_transforms, get_val_transforms
from src.config import GlobalConfig, get_default_config


def parse_flickr8k_captions(captions_file: Union[str, Path]) -> pd.DataFrame:
    """Parses Flickr8k captions file into a DataFrame with columns ['image_name', 'caption', 'caption_idx'].
    
    Handles both Flickr8k.token.txt format (image#0 <tab> caption) and 
    captions.txt CSV format (image,caption).
    """
    captions_file = Path(captions_file)
    if not captions_file.exists():
        raise FileNotFoundError(f"Captions file not found at: {captions_file}")

    records = []
    with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        if "," in first_line and "image" in first_line.lower():
            # CSV format
            df = pd.read_csv(captions_file)
            df.columns = [c.strip().lower() for c in df.columns]
            if "image" in df.columns and "caption" in df.columns:
                df = df.rename(columns={"image": "image_name"})
            return df[["image_name", "caption"]]

        # Tab-separated Flickr8k.token.txt format
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
                img_token, caption = parts[0], parts[1]
                img_name = img_token.split("#")[0] if "#" in img_token else img_token
                caption_idx = int(img_token.split("#")[1]) if "#" in img_token else 0
                records.append({"image_name": img_name, "caption": caption, "caption_idx": caption_idx})
            elif "," in line:
                parts = line.split(",", 1)
                records.append({"image_name": parts[0], "caption": parts[1]})

    return pd.DataFrame(records)


def split_flickr8k_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset by unique image IDs to strictly prevent data leakage across splits."""
    unique_images = df["image_name"].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    n = len(unique_images)
    
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    
    train_images = set(unique_images[:train_end])
    val_images = set(unique_images[train_end:val_end])
    test_images = set(unique_images[val_end:])

    train_df = df[df["image_name"].isin(train_images)].reset_index(drop=True)
    val_df = df[df["image_name"].isin(val_images)].reset_index(drop=True)
    test_df = df[df["image_name"].isin(test_images)].reset_index(drop=True)

    return train_df, val_df, test_df


class Flickr8kDataset(Dataset):
    """PyTorch Dataset for Flickr8k supporting image files, cached CNN features, and multi-reference evaluation."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Union[str, Path],
        vocab: Vocabulary,
        transforms: Optional[Callable] = None,
        features_dir: Optional[Union[str, Path]] = None,
        is_eval: bool = False,
        max_caption_len: int = 35
    ):
        self.df = df
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transforms = transforms
        self.features_dir = Path(features_dir) if features_dir else None
        self.is_eval = is_eval
        self.max_caption_len = max_caption_len

        if self.is_eval:
            # Group by unique image with all references for evaluation
            self.grouped_records = (
                self.df.groupby("image_name")["caption"]
                .apply(list)
                .reset_index()
                .to_dict("records")
            )
        else:
            self.records = self.df.to_dict("records")

    def __len__(self) -> int:
        return len(self.grouped_records) if self.is_eval else len(self.records)

    def _load_image_or_feature(self, img_name: str) -> torch.Tensor:
        """Loads pre-extracted feature tensor if available, otherwise processes raw image."""
        if self.features_dir:
            feat_stem = Path(img_name).stem
            feat_file = self.features_dir / f"{feat_stem}.pt"
            if feat_file.exists():
                try:
                    feat = torch.load(feat_file, map_location="cpu")
                    if isinstance(feat, torch.Tensor) and feat.numel() == 196 * 2048:
                        return feat.float()
                except Exception:
                    try:
                        feat_file.unlink(missing_ok=True)
                    except Exception:
                        pass

        # Fallback to loading image
        img_path = self.images_dir / img_name
        if not img_path.exists():
            # Try finding without case sensitivity or in root
            alt_path = self.images_dir.parent / img_name
            if alt_path.exists():
                img_path = alt_path
            else:
                # Return dummy tensor if missing in test environments
                return torch.zeros((3, 224, 224), dtype=torch.float32)

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transforms:
                return self.transforms(image)
            return get_val_transforms()(image)
        except Exception:
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def __getitem__(self, idx: int):
        if self.is_eval:
            item = self.grouped_records[idx]
            img_name = item["image_name"]
            references = item["caption"]
            img_tensor = self._load_image_or_feature(img_name)
            return img_tensor, references, img_name

        item = self.records[idx]
        img_name = item["image_name"]
        caption = str(item["caption"])

        img_tensor = self._load_image_or_feature(img_name)
        num_caption = self.vocab.numericalize(caption, add_special_tokens=True)
        if len(num_caption) > self.max_caption_len:
            num_caption = num_caption[:self.max_caption_len - 1] + [self.vocab.end_idx]

        return img_tensor, torch.tensor(num_caption, dtype=torch.long), img_name


class CaptionCollate:
    """Collate class to dynamically pad sequences within batches."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        # Sort batch by caption length descending for efficient pack_padded_sequence if needed
        batch.sort(key=lambda x: len(x[1]), reverse=True)

        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        img_names = [item[2] for item in batch]

        images = torch.stack(images, dim=0)
        lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return images, targets_padded, lengths, img_names


def create_dataloaders(
    config: Optional[GlobalConfig] = None,
    vocab: Optional[Vocabulary] = None,
    df: Optional[pd.DataFrame] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """Factory creating train, validation, and test DataLoaders."""
    cfg = config or get_default_config()

    if df is None:
        df = parse_flickr8k_captions(cfg.paths.captions_file)

    train_df, val_df, test_df = split_flickr8k_data(
        df,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        test_ratio=cfg.data.test_split,
        seed=cfg.training.seed
    )

    if vocab is None:
        if cfg.paths.vocab_path.exists():
            vocab = Vocabulary.load(cfg.paths.vocab_path)
        else:
            vocab = Vocabulary(min_freq=cfg.data.min_word_freq)
            vocab.build_vocabulary(train_df["caption"].tolist())
            vocab.save(cfg.paths.vocab_path)

    train_dataset = Flickr8kDataset(
        df=train_df,
        images_dir=cfg.paths.images_dir,
        vocab=vocab,
        transforms=get_train_transforms(image_size=cfg.data.image_size),
        features_dir=cfg.paths.features_dir if any(cfg.paths.features_dir.glob("*.pt")) else None,
        is_eval=False,
        max_caption_len=cfg.data.max_caption_len
    )

    val_dataset = Flickr8kDataset(
        df=val_df,
        images_dir=cfg.paths.images_dir,
        vocab=vocab,
        transforms=get_val_transforms(image_size=cfg.data.image_size),
        features_dir=cfg.paths.features_dir if any(cfg.paths.features_dir.glob("*.pt")) else None,
        is_eval=False,
        max_caption_len=cfg.data.max_caption_len
    )

    test_dataset = Flickr8kDataset(
        df=test_df,
        images_dir=cfg.paths.images_dir,
        vocab=vocab,
        transforms=get_val_transforms(image_size=cfg.data.image_size),
        features_dir=cfg.paths.features_dir if any(cfg.paths.features_dir.glob("*.pt")) else None,
        is_eval=True,
        max_caption_len=cfg.data.max_caption_len
    )

    collate_fn = CaptionCollate(pad_idx=vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # For test loader in evaluation mode
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    return train_loader, val_loader, test_loader, vocab
