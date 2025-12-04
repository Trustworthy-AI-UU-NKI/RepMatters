# U-Net without skip connections using UM-Net base blocks for the decoder = U-Net*

from pathlib import Path
import lightning.pytorch as pl
from unet_parts import *

# UM-Net base blocks
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4,  3, 1, 1),
                                   nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        return x3

# without skip connections
class UNet(pl.LightningModule):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)
        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(32, 2, kernel_size=1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.final(x)
        return logits
