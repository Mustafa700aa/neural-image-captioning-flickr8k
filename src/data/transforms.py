"""Image transformations and data augmentation pipelines."""

from typing import Tuple
from torchvision import transforms


def get_train_transforms(image_size: Tuple[int, int] = (224, 224), 
                         mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                         std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> transforms.Compose:
    """Returns training transforms pipeline with data augmentations."""
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.14), int(image_size[1] * 1.14))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224),
                       mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                       std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> transforms.Compose:
    """Returns deterministic validation and test image transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(image_size: Tuple[int, int] = (224, 224),
                            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> transforms.Compose:
    """Returns inference transform pipeline for single images."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
