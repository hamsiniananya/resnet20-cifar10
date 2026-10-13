
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ShortcutA(nn.Module):
    """CIFAR ResNet Option A shortcut."""

    def __init__(self, stride, in_channels, out_channels):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        if self.stride != 1:
            x = x[:, :, ::self.stride, ::self.stride]

        if self.out_channels > self.in_channels:
            diff = self.out_channels - self.in_channels
            x = F.pad(
                x,
                (0, 0, 0, 0, diff // 2, diff - diff // 2),
            )

        return x


class BasicBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = ShortcutA(
            stride, in_channels, out_channels
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        return F.relu(out)


class CifarNet20(nn.Module):
    """CIFAR-10 20-layer network, with optional skip connections."""

    def __init__(self, residual=True, num_classes=10):
        super().__init__()
        self.residual = residual

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)

        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, in_channels, out_channels, stride):
        blocks = []

        for i in range(3):
            current_stride = stride if i == 0 else 1
            current_in = in_channels if i == 0 else out_channels

            if self.residual:
                blocks.append(
                    BasicBlock(
                        current_in, out_channels, current_stride
                    )
                )
            else:
                # Same number of convolutions, no skip connections.
                blocks.extend([
                    conv3x3(current_in, out_channels, current_stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    conv3x3(out_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ])

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return self.fc(x)


def ResNet20():
    return CifarNet20(residual=True)


def PlainNet20():
    return CifarNet20(residual=False)
