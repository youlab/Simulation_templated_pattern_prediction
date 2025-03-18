import torch
import torch.nn as nn

class PDEArenaDilatedBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dilation_rates, activation=nn.ReLU, norm=True):
        super(PDEArenaDilatedBlock, self).__init__()
        self.dilated_layers = nn.ModuleList([
            nn.Conv2d(
                in_planes if i == 0 else out_planes,
                out_planes,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                bias=False
            )
            for i, rate in enumerate(dilation_rates)
        ])
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm2d(out_planes) if norm else nn.Identity()
            for _ in dilation_rates
        ])
        self.activation = activation(inplace=True)
        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes) if norm else nn.Identity()
            )

    def forward(self, x):
        out = x
        for layer, norm in zip(self.dilated_layers, self.norm_layers):
            out = self.activation(norm(layer(out)))
        return out + self.shortcut(x)

class PDEArenaDilatedResNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_blocks=15, dilation_rates=[1, 2, 4, 8], activation=nn.ReLU, norm=True):
        super(PDEArenaDilatedResNet, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.layers = nn.Sequential(
            *[PDEArenaDilatedBlock(hidden_channels, hidden_channels, dilation_rates, activation=activation, norm=norm)
              for _ in range(num_blocks)]
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        return self.out_conv(x)
