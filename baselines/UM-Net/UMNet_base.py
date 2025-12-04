import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)

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
    
class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SideoutBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x 
    
class UMNet(nn.Module):
    def __init__(self, num_classes):
        super(UMNet, self).__init__()
        resnet = models.resnet34(pretrained=True)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu  # 64
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64
        self.encoder3 = resnet.layer2  # 128
        self.encoder4 = resnet.layer3  # 256
        self.encoder5 = resnet.layer4  # 512

        # Downsampling with 1x1 convolution
        self.down3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # Decoder blocks
        self.decoder5 = DecoderBlock(in_channels=64, out_channels=64)
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder3 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)

        # Final output layer
        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1) # [4, 64, 112, 112]
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool) # 4, 64, 56, 56]
        e3 = self.encoder3(e2) # [4, 128, 28, 28]
        e4 = self.encoder4(e3) # [4, 256, 14, 14]
        e5 = self.encoder5(e4) # [4, 512, 7, 7]

        # Apply downsampling
        e3 = self.down3(e3)  # 64 - [4, 64, 28, 28]
        e4 = self.down4(e4)  # 64 - [4, 64, 14, 14]
        e5 = self.down5(e5)  # 64 - [4, 64, 7, 7]

        # Decoder path
        d5 = self.decoder5(e5) # [4, 64, 14, 14]
        
        d41 = torch.cat((d5, e4), dim=1)
        d4 = self.decoder4(d41) # [4, 64, 28, 28]

        d31 = torch.cat((d4, e3), dim=1)
        d3 = self.decoder3(d31) # [4, 64, 56, 56]

        d21 = torch.cat((d3, e2), dim=1)
        d2 = self.decoder2(d21) # [4, 64, 112, 112]

        # Final output layer
        out1 = self.final(d2)
        out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True) # [4, 2, 224, 224]
        
        return out1 
