import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CropBlock(nn.Module):
    def __init__(self, crop_size):
        super(CropBlock, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        _, _, h, w = x.size()
        start_idx = (h - self.crop_size) // 2
        end_idx = start_idx + self.crop_size
        x = x[:, :, start_idx:end_idx, start_idx:end_idx]
        return x


class FCDeepONet(nn.Module):
    def __init__(self):
        super(FCDeepONet, self).__init__()
        self.branch = nn.Sequential()
        self.branch.add_module('conv1', nn.Conv2d(2, 16, kernel_size=5, stride=2))
        self.branch.add_module('active1', Swish())
        self.branch.add_module('conv2', nn.Conv2d(16, 32, kernel_size=5, stride=2))
        self.branch.add_module('active2', Swish())
        self.branch.add_module('conv3', nn.Conv2d(32, 64, kernel_size=5, stride=2))
        self.branch.add_module('active3', Swish())
        self.branch.add_module('conv4', nn.Conv2d(64, 64, kernel_size=5, stride=2))
        self.branch.add_module('active4', Swish())
        self.branch.add_module('conv5', nn.Conv2d(64, 128, kernel_size=3, stride=1))
        self.branch.add_module('active5', Swish())
        self.branch.add_module('conv6', nn.Conv2d(128, 200, kernel_size=1))

        self.trunk = nn.Sequential()
        self.trunk.add_module('hidden1', nn.Linear(2, 200))
        self.trunk.add_module('norm1', nn.LayerNorm(200))
        self.trunk.add_module('active1', Swish())
        self.trunk.add_module('hidden2', nn.Linear(200, 200))
        self.trunk.add_module('active2', Swish())
        self.trunk.add_module('hidden3', nn.Linear(200, 200))
        self.trunk.add_module('active3', nn.ReLU())

        self.decoder = nn.Sequential()
        self.decoder.add_module('upconv1', UpConvBlock(200, 128, 3))
        self.decoder.add_module('conv1', ConvBlock(128, 128))
        self.decoder.add_module('upconv2', UpConvBlock(128, 64, 4))
        self.decoder.add_module('conv2', ConvBlock(64, 64))
        self.decoder.add_module('upconv3', UpConvBlock(64, 32, 2))
        self.decoder.add_module('conv3', ConvBlock(32, 32))
        self.decoder.add_module('upconv4', UpConvBlock(32, 16, 2))
        self.decoder.add_module('conv4', ConvBlock(16, 16))
        self.decoder.add_module('crop', CropBlock(140))
        self.decoder.add_module('conv5', nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, input_tra, Xs_val):
        topo = input_tra[:, 1:2, :, :]
        branch_out = self.branch(input_tra)
        trunk_out = self.trunk(Xs_val)
        decoder_in = torch.einsum("ij,ijkl->ijkl", trunk_out, branch_out)
        decoder_out = self.decoder(decoder_in)
        out = decoder_out*topo
        return out

