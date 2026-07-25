import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 32,
        features=None,
    ):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        # Encoder blocks
        self.downs = nn.ModuleList()

        # Decoder blocks
        self.ups = nn.ModuleList()

        # Shared pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        current_channels = in_channels

        for feature in features:
            self.downs.append(
                DoubleConv(current_channels, feature)
            )
            current_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2,
        )

        # Decoder
        for feature in reversed(features):

            # Upsampling
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )

            # Refinement after concatenation
            self.ups.append(
                DoubleConv(
                    feature * 2,
                    feature,
                )
            )

        # Pixel-wise classifier
        self.final_conv = nn.Conv2d(
            features[0],
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Deepest skip first
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):

            # Upsample
            x = self.ups[idx](x)

            # Matching skip connection
            skip = skip_connections[idx // 2]

            # Handle odd input resolutions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Concatenate encoder features
            x = torch.cat((skip, x), dim=1)

            # Refine features
            x = self.ups[idx + 1](x)

        # Final segmentation logits
        return self.final_conv(x)