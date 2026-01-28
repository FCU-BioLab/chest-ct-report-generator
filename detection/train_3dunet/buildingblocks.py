from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F


# ============ Attention Modules ============

class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation Block
    Channel attention mechanism for 3D volumes
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1, 1)
        # Scale
        return x * y.expand_as(x)


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate for skip connections
    Highlights relevant features while suppressing irrelevant regions
    """
    def __init__(self, gate_channels, in_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.GroupNorm(min(8, inter_channels), inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.GroupNorm(min(8, inter_channels), inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: gating signal from decoder (lower resolution)
            x: skip connection from encoder (higher resolution)
        """
        # Upsample g to match x's spatial size
        g_upsampled = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g_upsampled)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

def create_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    order: str,
    num_groups: int,
    padding: int | tuple[int],
    dropout_prob: float,
    is3d: bool,
) -> list[tuple[str, nn.Module]]:
    """
    Create a list of modules for a given level of UNet network.
    """
    assert "c" in order, "Conv layer MUST be present"
    assert order[0] not in "rle", "Non-linearity cannot be the first operation in the layer"

    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ("g" in order or "b" in order)
            if is3d:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

            modules.append(("conv", conv))
        elif char == "g":
            is_before_conv = i < order.index("c")
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, (
                f"Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}"
            )
            modules.append(("groupnorm", nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == "b":
            is_before_conv = i < order.index("c")
            if is3d:
                bn = nn.BatchNorm3d
            else:
                bn = nn.BatchNorm2d

            if is_before_conv:
                modules.append(("batchnorm", bn(in_channels)))
            else:
                modules.append(("batchnorm", bn(out_channels)))
        elif char == "d":
            modules.append(("dropout", nn.Dropout(p=dropout_prob)))
        elif char == "D":
            modules.append(("dropout2d", nn.Dropout2d(p=dropout_prob)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'd', 'D']"
            )

    return modules


class SingleConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        for name, module in create_conv(
            in_channels, out_channels, kernel_size, order, num_groups, padding, dropout_prob, is3d
        ):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        encoder,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        upscale=2,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            if upscale == 1:
                conv1_out_channels = out_channels
            else:
                conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        if isinstance(dropout_prob, list) or isinstance(dropout_prob, tuple):
            dropout_prob1 = dropout_prob[0]
            dropout_prob2 = dropout_prob[1]
        else:
            dropout_prob1 = dropout_prob2 = dropout_prob

        self.add_module(
            "SingleConv1",
            SingleConv(
                conv1_in_channels,
                conv1_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                dropout_prob=dropout_prob1,
                is3d=is3d,
            ),
        )
        self.add_module(
            "SingleConv2",
            SingleConv(
                conv2_in_channels,
                conv2_out_channels,
                kernel_size,
                order,
                num_groups,
                padding=padding,
                dropout_prob=dropout_prob2,
                is3d=is3d,
            ),
        )


class ResNetBlock(nn.Module):
    """Residual block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, order="cge", num_groups=8, is3d=True, **kwargs):
        super().__init__()

        if in_channels != out_channels:
            if is3d:
                self.conv1 = nn.Conv3d(in_channels, out_channels, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv1 = nn.Identity()

        self.conv2 = SingleConv(
            out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups, is3d=is3d
        )
        n_order = order
        for c in "rel":
            n_order = n_order.replace(c, "")
        self.conv3 = SingleConv(
            out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups, is3d=is3d
        )

        if "l" in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif "e" in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        apply_pooling=True,
        pool_kernel_size=2,
        pool_type="max",
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upscale=2,
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()
        if apply_pooling:
            if pool_type == "max":
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=True,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            upscale=upscale,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        scale_factor=2,
        basic_module=DoubleConv,
        conv_layer_order="gcr",
        num_groups=8,
        padding=1,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super().__init__()

        concat = True
        adapt_channels = False

        if upsample is not None and upsample != "none":
            if upsample == "default":
                if basic_module == DoubleConv:
                    upsample = "nearest"
                    concat = True
                    adapt_channels = False
                elif basic_module == ResNetBlock:
                    upsample = "deconv"
                    concat = False
                    adapt_channels = True

            if upsample == "deconv":
                self.upsampling = TransposeConvUpsampling(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size,
                    scale_factor=scale_factor,
                    is3d=is3d,
                )
            else:
                self.upsampling = InterpolateUpsampling(mode=upsample)
        else:
            self.upsampling = NoUpsampling()
            concat = True

        self.joining = partial(self._joining, concat=concat)

        if adapt_channels is True:
            in_channels = out_channels

        self.basic_module = basic_module(
            in_channels,
            out_channels,
            encoder=False,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            padding=padding,
            dropout_prob=dropout_prob,
            is3d=is3d,
        )

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class AbstractUpsampling(nn.Module):
    def __init__(self, upsample):
        super().__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    def __init__(self, mode="nearest"):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    class Upsample(nn.Module):
        def __init__(self, conv_transposed, is3d):
            super().__init__()
            self.conv_transposed = conv_transposed
            self.is3d = is3d

        def forward(self, x, size):
            x = self.conv_transposed(x)
            return F.interpolate(x, size=size)

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, is3d=True):
        if is3d:
            conv_transposed = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1, bias=False
            )
        else:
            conv_transposed = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1, bias=False
            )
        upsample = self.Upsample(conv_transposed, is3d)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x

def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, conv_upscale, dropout_prob, layer_order, num_groups, pool_kernel_size, is3d):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(
                in_channels, out_feature_num, apply_pooling=False, basic_module=basic_module,
                conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size, num_groups=num_groups,
                padding=conv_padding, upscale=conv_upscale, dropout_prob=dropout_prob, is3d=is3d
            )
        else:
            encoder = Encoder(
                f_maps[i - 1], out_feature_num, basic_module=basic_module,
                conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size, num_groups=num_groups,
                pool_kernel_size=pool_kernel_size, padding=conv_padding, upscale=conv_upscale,
                dropout_prob=dropout_prob, is3d=is3d
            )
        encoders.append(encoder)
    return nn.ModuleList(encoders)

def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample, dropout_prob, is3d):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv and upsample != "deconv":
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]
        out_feature_num = reversed_f_maps[i + 1]
        
        decoder = Decoder(
            in_feature_num, out_feature_num, basic_module=basic_module,
            conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size, num_groups=num_groups,
            padding=conv_padding, upsample=upsample, dropout_prob=dropout_prob, is3d=is3d
        )
        decoders.append(decoder)
    return nn.ModuleList(decoders)
