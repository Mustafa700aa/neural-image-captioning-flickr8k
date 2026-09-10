"""CNN Vision Encoders for feature extraction using Transfer Learning."""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Pretrained CNN Vision Encoder extracting spatial feature maps.
    
    Supports ResNet-50, MobileNetV3-Large, and EfficientNet-B0 backbones.
    Retains 2D spatial feature grid (e.g. 14x14 or 7x7) for spatial attention mechanisms.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        encoded_image_size: int = 14,
        fine_tune: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.encoded_image_size = encoded_image_size

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            cnn = models.resnet50(weights=weights)
            # Remove adaptive avg pool and linear classifier
            modules = list(cnn.children())[:-2]
            self.cnn = nn.Sequential(*modules)
            self.feature_dim = 2048
        elif self.backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            cnn = models.mobilenet_v3_large(weights=weights)
            self.cnn = cnn.features
            self.feature_dim = 960
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            cnn = models.efficientnet_b0(weights=weights)
            self.cnn = cnn.features
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone}")

        # Adaptive pool to ensure exact spatial dimensions (e.g. 14x14)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        self.fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts spatial visual feature maps from input images.
        
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            
        Returns:
            features: Tensor of shape (batch_size, num_pixels, feature_dim), where num_pixels = encoded_image_size^2
        """
        # If images already extracted features (shape: B, num_pixels, dim or B, dim, H, W)
        if images.dim() == 3:
            return images

        out = self.cnn(images)  # (batch_size, feature_dim, H', W')
        out = self.adaptive_pool(out)  # (batch_size, feature_dim, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, feature_dim)
        out = out.reshape(out.size(0), -1, self.feature_dim)  # (batch_size, num_pixels, feature_dim)
        return out

    def fine_tune(self, fine_tune: bool = True) -> None:
        """Enables or disables gradient computation for the CNN backbone."""
        for p in self.cnn.parameters():
            p.requires_grad = False

        if fine_tune:
            # Only fine-tune the deeper convolutional blocks
            if self.backbone_name == "resnet50":
                for c in list(self.cnn.children())[6:]:
                    for p in c.parameters():
                        p.requires_grad = True
            elif self.backbone_name in ("mobilenet_v3_large", "efficientnet_b0"):
                for c in list(self.cnn.children())[10:]:
                    for p in c.parameters():
                        p.requires_grad = True
