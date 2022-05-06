import math
import torch
from torch import nn 
import torch.nn.functional as F

# ------------------------------------------------------------
#                            UNet
# ------------------------------------------------------------
# This code is adapted from: https://github.com/ELEKTRONN/elektronn3/blob/master/elektronn3/models/unet.py
# The corresponding paper is:
# Olaf Ronneberger, Philipp Fischer, and Thomas Brox
# U-net: Convolutional networks for biomedical image segmentation
# International Conference on Medical Image Computing and Computer-Assisted Intervention, 2015
# Available from: https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

def get_normalization(channels, normalization, dims):
    """Returns the appropriate normalization layer given channels, dims.
    Arguments:
        channels: number of layer channels for normalization
        normalization: desired normalization layer (str)
        dims: data dimensions (1D, 2D, or 3D)
    
    Returns:
        normalization layer
    """
    if normalization == 'BatchNorm':
        if dims == 1:
            return nn.BatchNorm1d(channels)
        elif dims == 2:
            return nn.BatchNorm2d(channels)
        elif dims == 3:
            return nn.BatchNorm3d(channels)
    elif normalization == 'InstanceNorm':
        if dims == 1:
            return nn.InstanceNorm1d(channels)
        elif dims == 2:
            return nn.InstanceNorm2d(channels)
        elif dims == 3:
            return nn.InstanceNorm3d(channels)
    elif normalization == 'LayerNorm':
        return nn.LayerNorm(channels)
    elif normalization == 'GroupNorm':
        return nn.GroupNorm(channels/4, channels)
    else:
        return None
    
def get_pooling(pooling, size, stride = None, padding = 0, dilation = 1, dims = 2):
    """Returns the appropriate pooling layer given pooling, dims.
    Arguments:
        pooling: desired pooling layer (str)
        size: kernel size or output size
        dims: data dimensions (1D, 2D, or 3D)
    
    Returns:
        pooling layer
    """
    if pooling == 'MaxPool':
        if dims == 1:
            return nn.MaxPool1d(size, stride = stride, padding = padding, dilation = dilation)
        elif dims == 2:
            return nn.MaxPool2d(size, stride = stride, padding = padding, dilation = dilation)
        elif dims == 3:
            return nn.MaxPool3d(size, stride = stride, padding = padding, dilation = dilation)
    elif pooling == 'AvgPool':
        if dims == 1:
            return nn.AvgPool1d(size, stride = stride, padding = padding)
        elif dims == 2:
            return nn.AvgPool2d(size, stride = stride, padding = padding)
        elif dims == 3:
            return nn.AvgPool3d(size, stride = stride, padding = padding)
    elif pooling == 'AdaptiveAvgPool':
        if dims == 1:
            return nn.AdaptiveAvgPool1d(size)
        if dims == 2:
            return nn.AdaptiveAvgPool2d(size)
        if dims == 3:
            return nn.AdaptiveAvgPool3d(size)
    else:
        return None

class BasicConv(nn.Module):
    """Basic convolution layer."""
    def __init__(self, dims, channels_in, channels_out, kernel_size = 3, 
                stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, 
                normalization = 'BatchNorm', activation = nn.ReLU(True), act_first = True):
        super(BasicConv, self).__init__()
        self.dims = dims

        if dims == 1:
            basic_conv = [nn.Conv1d(channels_in, channels_out, kernel_size = kernel_size, 
                                    stride = stride, padding = padding, dilation = dilation, 
                                    groups = groups, bias = bias)]
        elif dims == 2:
            basic_conv = [nn.Conv2d(channels_in, channels_out, kernel_size = kernel_size, 
                                    stride = stride, padding = padding, dilation = dilation, 
                                    groups = groups, bias = bias)]
        else: #dims == 3:
            basic_conv = [nn.Conv3d(channels_in, channels_out, kernel_size = kernel_size, 
                                    stride = stride, padding = padding, dilation = dilation, 
                                    groups = groups, bias = bias)]

        if activation and act_first: 
            basic_conv.append(activation)
        if normalization and normalization != 'None': 
            basic_conv.append(get_normalization(channels_out, normalization, dims))
        if activation and not act_first:
            basic_conv.append(activation)
        self.body = nn.Sequential(*basic_conv)

    def forward(self, x):
        return self.body(x)

class UNetConv(nn.Module):
    """UNet convolution layer"""
    def __init__(self, dims, in_channels, out_channels, mid_channels = None, normalization = 'BatchNorm', activation = nn.ReLU(True), res = False):
        super().__init__()
        self.res = res

        if not mid_channels:
            mid_channels = out_channels

        self.initial_conv = BasicConv(dims, in_channels, mid_channels, kernel_size = 3, padding = 1, normalization = normalization, activation = activation)

        res_conv = []
        for _ in range(2):
            res_conv.append(BasicConv(dims, mid_channels, mid_channels, kernel_size = 3, padding = 1, normalization = normalization, activation = activation))
        self.res_conv = nn.Sequential(*res_conv)

        self.final_conv = BasicConv(dims, mid_channels, out_channels, kernel_size = 3, padding = 1, normalization = normalization, activation = activation)

    def forward(self, x):
        x = self.initial_conv(x)

        if self.res:
            res = self.res_conv(x)
            res += x
            out = self.final_conv(res)
        else:
            out = self.final_conv(x)
        return out

class Down(nn.Module):
    """UNet contracting layer"""
    def __init__(self, dims, in_channels, out_channels, normalization, activation, res = False):
        super().__init__()

        maxpool_conv = []
        maxpool_conv.append(get_pooling('MaxPool', size = 2, dims = dims))
        maxpool_conv.append(UNetConv(dims, in_channels, out_channels, None, normalization, activation, res))
        self.maxpool_conv = nn.Sequential(*maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """UNet expanding layer"""
    def __init__(self, dims, in_channels, out_channels, normalization, activation, bilinear = True, res = False):
        super().__init__()
        self.dims = dims

        if self.dims == 1 or self.dims == 3:
            self.up = nn.Upsample(scale_factor=2)
            self.conv = UNetConv(dims, in_channels, out_channels, in_channels // 2, normalization, activation, res)
        elif self.dims == 2:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = UNetConv(dims, in_channels, out_channels, in_channels // 2, normalization, activation, res)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = UNetConv(dims, in_channels, out_channels, None, normalization, activation, res)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.dims == 1:
            diffY = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        elif self.dims == 2:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        elif self.dims == 3:
            diffZ = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffX = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                           diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, dims, channels, img_channels = 3, num_classes = 0, n_blocks = 3, normalization = 'BatchNorm', activation = nn.ReLU(), bilinear = True, res = False):
        super(UNet, self).__init__()
        self.dims = dims
        self.channels = channels
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.n_blocks = n_blocks
        self.normalization = normalization
        self.activation = activation
        self.bilinear = bilinear
        self.res = res
        
        if self.dims == 1 or self.channels <= 3:
            self.filter_config=(64, 128, 256, 512, 1024)
        else:
            self.filter_config=(self.channels, self.channels*2, self.channels*4, self.channels*8, self.channels*16)
            
        factor = 2 if bilinear else 1

        self.upsample = None
        self.inc = UNetConv(self.dims, channels, self.filter_config[0], None, self.normalization, self.activation, self.res)
        
        down_blocks = []
        up_blocks = []
        
        for i in range(n_blocks-1):
            down_blocks.append(Down(self.dims, self.filter_config[i], self.filter_config[i+1], self.normalization, self.activation, self.res))
            up_blocks.append(Up(self.dims, self.filter_config[n_blocks-i], self.filter_config[n_blocks-i-1] // factor, self.normalization, self.activation, self.bilinear, self.res))
        down_blocks.append(Down(self.dims, self.filter_config[n_blocks-1], self.filter_config[n_blocks] // factor, self.normalization, self.activation, self.res))
        up_blocks.append(Up(self.dims, self.filter_config[1], self.filter_config[0], self.normalization, self.activation, self.bilinear, self.res))

        self.down_blocks = nn.Sequential(*down_blocks)
        self.up_blocks = nn.Sequential(*up_blocks)

        self.out = None
        self.avgpool = None
        self.tail = None
        
        if self.dims == 1:
            self.out = BasicConv(self.dims, self.filter_config[0], 1, padding = 1, normalization = self.normalization, activation = self.activation)
        elif self.dims == 2:
            self.tail = BasicConv(self.dims, self.filter_config[0], self.num_classes, kernel_size = 1, normalization = None, activation = None)
        elif self.dims == 3:
            self.out = BasicConv(self.dims, self.filter_config[0], 1, padding = 1, normalization = self.normalization, activation = self.activation)
            self.tail = BasicConv(2, self.img_channels, self.num_classes, kernel_size = 1, normalization = None, activation = None)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.inc(x)
        
        encoder_output = [x]
        
        for module in self.down_blocks:
            x = module(x)
            encoder_output.append(x)
                    
        for i, module in enumerate(self.up_blocks):
            x = module(x, encoder_output[-(i+2)])
            
        if self.out is not None:
            x = self.out(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        if self.dims == 3:
            x = torch.flatten(x,1,2)
        if self.tail is not None:
            x = self.tail(x)
        return x
