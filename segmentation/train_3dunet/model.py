#!/usr/bin/env python3
"""
3D U-Net Implementation (Refactored)
====================================
Based on wolny/pytorch-3dunet
"""

import torch.nn as nn
from .buildingblocks import (
    DoubleConv, ResNetBlock, create_encoders, create_decoders, NoUpsampling
)

class UNet3D(nn.Module):
    """
    3D U-Net model with flexible configuration.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        f_maps=64,
        layer_order='gcr',
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_padding=1,
        conv_kernel_size=3,
        conv_upscale=2,
        dropout_prob=0.1,
        pool_kernel_size=2,
        basic_module=DoubleConv,
        testing=False,
    ):
        super().__init__()

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2 ** k for k in range(num_levels)]

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # Create Encoders
        self.encoders = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, conv_upscale, dropout_prob,
            layer_order, num_groups, pool_kernel_size, is3d=True
        )

        # Create Decoders
        self.decoders = create_decoders(
            f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample="default",
            dropout_prob=dropout_prob, is3d=True
        )

        # Final Convolution
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation and not testing:
            self.final_activation = nn.Identity() # Sigmoid/Softmax handled in Loss or Training Loop usually
        else:
            self.final_activation = nn.Identity()

    def forward(self, x):
        # Encoder
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder processing
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Decoder
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

def get_model(config) -> nn.Module:
    """Get model from config"""
    # Defaults or Config
    f_maps = [32, 64, 128, 256] # Standard base 32, 4 levels
    # Optional: check if config has these values
    if hasattr(config.model, 'f_maps'):
        f_maps = config.model.f_maps
    
    layer_order = 'gcr' # GroupNorm -> Conv -> ReLU (Standard in wolny/3dunet)
    
    return UNet3D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        f_maps=f_maps,
        layer_order=layer_order,
        basic_module=DoubleConv # or ResNetBlock
    )

if __name__ == "__main__":
    import torch
    # Test
    model = UNet3D(in_channels=1, out_channels=1, f_maps=[16, 32, 64, 128])
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    x = torch.randn(1, 1, 32, 128, 128)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
