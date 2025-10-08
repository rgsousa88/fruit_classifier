import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import math

from posencoding import *

def model_factory(model_name:str, num_classes:int, pretrained=True):
    supported_models = ['resnet50', 'resnet18',
                        'mobilenet_v3_large', 'mobilenet_v3_small',
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                        'vit_b_16','vit_b_32',
                        'swin_t','swin_b', 'swin_v2_t','swin_v2_s',
                        'custom_ViT_PositionalEncoding']
    
    if not model_name in supported_models:
        raise Exception("Invalid model name {model_name}")
    
    if model_name == 'custom_ViT_PositionalEncoding':
        model = ViTWithComplexPositionalEncoding(image_size=224, num_classes=num_classes, dim=512)
    else:
        weights = 'DEFAULT' if pretrained else None
        model = models.get_model(model_name, weights=weights)
    
    if 'resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'mobilenet' in model_name or 'efficientnet' in model_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'vit_b' in model_name:
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
    elif 'swin_' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)

    return model

class ResNetPatchEmbedding(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        patch_size: int = 16,
        output_dim: int = 512,
        pretrained: bool = True,
        use_avg_pool: bool = False,
        use_projection: bool = True
    ):
        """
        ResNet-based patch embedding extractor for transformer models.
        
        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', etc.)
            patch_size: Size of patches to extract features from
            output_dim: Fixed output dimension for embeddings (default: 512)
            pretrained: Whether to use pretrained weights
            use_avg_pool: Whether to use average pooling on features
            use_projection: Whether to project features to output_dim
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.use_avg_pool = use_avg_pool
        self.use_projection = use_projection
        
        # Load pretrained ResNet backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = getattr(models, backbone)(weights=weights)
        
        # Remove classification head and keep feature extractor
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze backbone if desired (optional)
        self.feature_extractor.requires_grad_(False)
        
        # Get the number of output channels from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.feature_extractor(dummy_input)
            self.in_channels = dummy_output.shape[1]
        
        # Adaptive average pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) if use_avg_pool else nn.Identity()
        
        # Projection layer to fixed output dimension
        if use_projection:
            self.projection = nn.Linear(self.in_channels, output_dim)
        else:
            self.projection = nn.Identity()
            if self.in_channels != output_dim:
                raise ValueError(f"Backbone output channels ({self.in_channels}) must match output_dim ({output_dim}) when use_projection=False")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patch embeddings from input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            embeddings: Patch embeddings of shape (B, num_patches, output_dim)
            grid_size: Tuple of (num_patches_h, num_patches_w)
        """
        batch_size, _, height, width = x.shape
        
        # Extract features using ResNet backbone
        features = self.feature_extractor(x)  # (B, C_feat, H_feat, W_feat)
        
        # Calculate the feature map dimensions
        feat_height, feat_width = features.shape[2], features.shape[3]
        
        # Calculate patch grid size
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        grid_size = (num_patches_h, num_patches_w)
        
        # If feature map doesn't match patch grid, we need to interpolate
        if feat_height != num_patches_h or feat_width != num_patches_w:
            features = F.interpolate(
                features, 
                size=(num_patches_h, num_patches_w), 
                mode='bilinear', 
                align_corners=False)
        
        # Apply adaptive average pooling if enabled
        if self.use_avg_pool:
            features = self.adaptive_pool(features)  # (B, C_feat, 1, 1)
            features = features.view(batch_size, self.in_channels, -1)  # (B, C_feat, num_patches)
        else:
            # Reshape to (B, C_feat, num_patches)
            features = features.view(batch_size, self.in_channels, -1)
        
        # Permute to (B, num_patches, C_feat)
        features = features.permute(0, 2, 1)
        
        # Project to fixed output dimension
        embeddings = self.projection(features)  # (B, num_patches, output_dim)
        
        return embeddings, grid_size
    
    def get_num_patches(self, height: int, width: int) -> int:
        """Calculate the number of patches for given image dimensions."""
        return (height // self.patch_size) * (width // self.patch_size)
    
    def get_positional_embeddings(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate sinusoidal positional embeddings for the patch grid.
        
        Args:
            grid_size: Tuple of (num_patches_h, num_patches_w)
            
        Returns:
            pos_embeddings: Positional embeddings of shape (num_patches, output_dim)
        """
        num_patches_h, num_patches_w = grid_size
        num_patches = num_patches_h * num_patches_w
        
        # Create grid positions
        positions = torch.arange(num_patches).unsqueeze(1)
        positions_h = (positions // num_patches_w).float()
        positions_w = (positions % num_patches_w).float()
        
        # Generate sinusoidal encodings
        div_term = torch.exp(torch.arange(0, self.output_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / self.output_dim))
        
        pos_embeddings = torch.zeros(num_patches, self.output_dim)
        pos_embeddings[:, 0::2] = torch.sin(positions_h * div_term.unsqueeze(0))
        pos_embeddings[:, 1::2] = torch.cos(positions_w * div_term.unsqueeze(0))
        
        return pos_embeddings

class ResNetPatchEmbeddingWithCLS(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        patch_size: int = 16,
        output_dim: int = 512,
        pretrained: bool = True,
        learnable_cls_token: bool = True
    ):
        """
        ResNet patch embedding extractor with optional [CLS] token.
        """
        super().__init__()
        
        self.patch_embedding = ResNetPatchEmbedding(
            backbone=backbone,
            patch_size=patch_size,
            output_dim=output_dim,
            pretrained=pretrained
        )
        
        # Learnable [CLS] token
        if learnable_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        else:
            self.register_buffer('cls_token', torch.zeros(1, 1, output_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patch embeddings and prepend [CLS] token.
        
        Returns:
            embeddings: Embeddings with shape (B, num_patches + 1, output_dim)
            grid_size: Tuple of (num_patches_h, num_patches_w)
        """
        patch_embeddings, grid_size = self.patch_embedding(x)
        batch_size = patch_embeddings.shape[0]
        
        # Expand cls token to batch size and prepend to patch embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        return embeddings, grid_size

# Example usage and test
if __name__ == "__main__":
    # Test the module
    embedder = ResNetPatchEmbedding(
        backbone="resnet50",
        patch_size=16,
        output_dim=512,
        pretrained=True
    )
    
    # Create dummy input (assumes preprocessing is done by dataloader)
    batch_size, channels, height, width = 2, 3, 224, 224
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    embeddings, grid_size = embedder(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Grid size: {grid_size}")
    print(f"Number of patches: {grid_size[0] * grid_size[1]}")
    
    # Test with CLS token
    embedder_with_cls = ResNetPatchEmbeddingWithCLS()
    embeddings_with_cls, grid_size = embedder_with_cls(dummy_input)
    print(f"Embeddings with CLS shape: {embeddings_with_cls.shape}")