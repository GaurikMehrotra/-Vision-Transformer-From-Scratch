"""A CIFAR ResNet-20 baseline: 3 stages, 3 residual blocks per stage."""

from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """[B,C,H,W] -> [B,Cout,H/stride,W/stride], with projection if needed."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Add the residual path and shortcut, then apply ReLU."""
        return self.act(self.main(x) + self.skip(x))


class ResNetSmall(nn.Module):
    """[B,3,32,32] -> [B,K], global average pooling before classifier."""

    def __init__(self, num_classes: int = 10, width: int = 16, blocks_per_stage: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        layers = []
        channels = width
        for stage in range(3):
            out = width * 2**stage
            for index in range(blocks_per_stage):
                stride = 2 if stage > 0 and index == 0 else 1
                layers.append(ResidualBlock(channels, out, stride))
                channels = out
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        """[B,3,H,W] -> [B,4*width,H/4,W/4] -> [B,K]."""
        return self.head(self.layers(self.stem(x)).mean(dim=(2, 3)))
