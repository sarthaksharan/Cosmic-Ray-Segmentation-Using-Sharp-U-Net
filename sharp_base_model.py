import torch
import torch.nn as nn
from torch import sigmoid
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting):
        super(type(self), self).__init__()

        # group norm fixed/dynamic group numbers
        group, channel, no_affine = norm_setting
        affine = not no_affine

        if group == 0 and channel > 0:
            # dynamically divide groups based on channel number
            n_group = max(1, int(out_ch // channel))
        else:
            # fixed group number
            n_group = group

        norms = nn.ModuleDict(
            {
                "batch": nn.BatchNorm2d(out_ch, momentum=0.005),
                "group": nn.GroupNorm(n_group, out_ch, affine=affine),
                "instance": nn.InstanceNorm2d(out_ch, affine=affine),
            }
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norms[norm],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norms[norm],
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
        
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting):
        super(type(self), self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, norm, norm_setting)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting, conv_type, down_type):
        super(type(self), self).__init__()

        downs = nn.ModuleDict(
            {
                "maxpool": nn.MaxPool2d(2),
                "avgpool": nn.AvgPool2d(2),
                "stride": nn.Conv2d(in_ch, in_ch, 3, 2, padding=1),
            }
        )

        convs = nn.ModuleDict(
            {
                "unet": DoubleConv(in_ch, out_ch, norm, norm_setting),
            }
        )

        self.down_cov = nn.Sequential(downs[down_type], convs[conv_type])

    def forward(self, x):
        x = self.down_cov(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm, norm_setting, up_type):
        super(type(self), self).__init__()

        if up_type == "deconv":
            self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        elif up_type == "upscale":
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            )

        else:
            raise ValueError(
                print("unknown up_type {up_type}, acceptable transconv, upscale")
            )

        self.conv = DoubleConv(in_ch, out_ch, norm, norm_setting)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(type(self), self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

def build_sharp_blocks(layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    in_channels = layer
    # Get kernel
    w = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
    
    # Change dimension
    w = np.expand_dims(w, axis=0)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=0)
    # Expand dimension
    w = np.expand_dims(w, axis=1)
    w = torch.Tensor(w)
    return w

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, padding = 1, bias=False):
        super(type(self), self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias)
        self.depthwise.weight = torch.nn.Parameter(build_sharp_blocks(in_ch))

    def forward(self, x):
        out = self.depthwise(x)
        return out+x

class UNet_module(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        hidden,
        norm,
        norm_setting,
        conv_type,
        down_type,
        up_type,
        sharp,
        deeper,
    ):
        super(UNet_module, self).__init__()
        self.deeper = deeper
        self.sharp = sharp

        self.inc = InConv(n_channels, hidden, norm, norm_setting)
        
        if sharp:
            self.sep1 = depthwise_separable_conv(hidden, hidden)
            self.sep1.depthwise.weight.requires_grad = False

        self.down1 = Down(hidden, hidden * 2, norm,norm_setting, conv_type, down_type)

        if deeper:
            self.down2 = Down(hidden * 2, hidden * 4, norm, norm_setting, conv_type, down_type)

            if sharp:
                self.sep2 = depthwise_separable_conv(hidden*2, hidden*2, f_param)
                self.sep2.depthwise.weight.requires_grad = False

            self.up2 = Up(hidden * 4, hidden * 2, norm, norm_setting, up_type)

        self.up1 = Up(hidden * 2, hidden , norm, norm_setting, up_type) 
        self.outc = OutConv(hidden, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        
        if self.sharp:
            x1 = self.sep1(x1)
        
        x2 = self.down1(x1)

        if self.deeper:
            x3 = self.down2(x2)
            x = self.up2(x3, x2)
            x = self.up1(x, x1)
        else:
            x = self.up1(x2, x1)

        x = self.outc(x)
        x = sigmoid(x)
        return x
                        
