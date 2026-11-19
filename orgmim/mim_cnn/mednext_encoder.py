"""MedNeXt-v1 encoder for OrgMIM, adapted from MIC-DKFZ/MedNeXt.

Copyright German Cancer Research Center (DKFZ), Division of Medical Image Computing.
See LICENSE-MedNeXt for the Apache-2.0 license.
"""

from torch import nn


# The encoder uses the first five stages of each MedNeXt-v1 configuration.
_SCALES = {
    "small": ((2, 2, 2, 2, 2), (2, 2, 2, 2, 2)),
    "base": ((2, 2, 2, 2, 2), (2, 3, 4, 4, 4)),
    "medium": ((3, 4, 4, 4, 4), (2, 3, 4, 4, 4)),
    "large": ((3, 4, 8, 8, 8), (3, 4, 8, 8, 8)),
}


class _MedNeXtBlock(nn.Module):
    """Depthwise convolution followed by channel expansion and compression."""

    def __init__(self, in_channels, out_channels, expansion, kernel_size, stride=1):
        super().__init__()
        expanded_channels = expansion * in_channels
        self.conv1 = nn.Conv3d(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=kernel_size // 2, groups=in_channels,
        )
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.conv2 = nn.Conv3d(in_channels, expanded_channels, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(expanded_channels, out_channels, 1)
        self.res_conv = (
            nn.Conv3d(in_channels, out_channels, 1, stride=stride)
            if stride != 1 else nn.Identity()
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(self.norm(y)))
        return self.conv3(y) + self.res_conv(x)


class MedNeXtEncoder(nn.Module):
    """3D encoder with five feature resolutions, from full size to 1/16."""

    def __init__(self, scale="base", in_channels=1, kernel_size=3):
        super().__init__()
        if scale not in _SCALES:
            raise ValueError(f"Unknown MedNeXt scale {scale!r}; choose from {', '.join(_SCALES)}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("MedNeXt kernel_size must be a positive odd integer")

        depths, expansions = _SCALES[scale]
        widths = self.get_feature_map_channels()
        self.stem = nn.Conv3d(in_channels, widths[0], 1)
        # These names match the MedNeXt encoder checkpoints exported by OrgMIM.
        for stage, (width, depth, expansion) in enumerate(zip(widths, depths, expansions)):
            name = f"enc_block_{stage}" if stage < 4 else "bottleneck"
            setattr(self, name, nn.Sequential(*[
                _MedNeXtBlock(width, width, expansion, kernel_size)
                for _ in range(depth)
            ]))
            if stage < 4:
                setattr(self, f"down_{stage}", _MedNeXtBlock(
                    width, widths[stage + 1], expansions[stage + 1], kernel_size, stride=2,
                ))

    def forward(self, x, hierarchical=False):
        x = self.stem(x)
        features = []
        for stage in range(4):
            x = getattr(self, f"enc_block_{stage}")(x)
            features.append(x)
            x = getattr(self, f"down_{stage}")(x)
        x = self.bottleneck(x)
        features.append(x)
        return features if hierarchical else x

    @staticmethod
    def get_downsample_ratio():
        return 16

    @staticmethod
    def get_feature_map_channels():
        return [32, 64, 128, 256, 512]
