# Basic components
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Models.replknetleaky import *

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.InstanceNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                              dtype=None),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.InstanceNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                              dtype=None),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_convNB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_convNB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconvNB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconvNB, self).__init__()
        self.conv = double_convNB(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(3, 2, 1),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.Droup = nn.Dropout(p=0.1,inplace=False)

    def forward(self, x):
        x = self.conv(x)
        # x = self.Droup(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNetS(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNetS, self).__init__()
        self.inc = inconv(n_channels, features)
        self.down11 = down(features, features * 2)
        self.down21 = down(features * 2, features * 4)
        self.down31 = down(features * 4, features * 8)
        self.down41 = down(features * 8, features * 16)
        self.up11 = up(features * 16, features * 8)
        self.up21 = up(features * 8, features * 4)
        self.up31 = up(features * 4, features * 2)
        self.up41 = up(features * 2, features)
        self.unet_1st_out = outconv(features, n_classes)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)
        x = self.up11(x5, x4)
        x = self.up21(x, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        return x


class LConvNeto(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(LConvNeto, self).__init__()
        self.inc = ReparamLargeKernelConv(in_channels=n_channels, out_channels=features, kernel_size=17, stride=1,
                                          groups=1, small_kernel=5)
        self.conv = ReparamLargeKernelConv(in_channels=features, out_channels=features, kernel_size=15, stride=1,
                                           groups=1, small_kernel=5)
        self.outc = ReparamLargeKernelConv(in_channels=features, out_channels=n_classes, kernel_size=13, stride=1,
                                           groups=1, small_kernel=5)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(features)
        self.normout = nn.InstanceNorm2d(n_classes)
        self.sconv = double_conv(features, features)

    def forward(self, x):
        x = self.inc(x)
        x = self.relu(x)
        x = self.norm(x)
        x1 = self.conv(x)
        x1 = self.relu(x1)
        x1 = self.norm(x1)
        x = x1 + x
        x1 = self.sconv(x)
        x = x1 + x
        x1 = self.sconv(x)
        x = x1 + x
        x1 = self.conv(x)
        x1 = self.relu(x1)
        x1 = self.norm(x1)
        x = x1 + x
        x1 = self.sconv(x)
        x = x1 + x
        x1 = self.sconv(x)
        x = x1 + x
        x = self.outc(x)
        x = self.relu(x)
        x = self.normout(x)
        return x


class LConvNet(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(LConvNet, self).__init__()
        self.inc = double_conv(n_channels, features)
        self.conv = ReparamLargeKernelConv(in_channels=features, out_channels=features, kernel_size=15, stride=1,
                                           groups=1, small_kernel=5)
        self.outc = double_conv(features, n_classes)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(features)
        self.normout = nn.InstanceNorm2d(n_classes)
        self.sconv = double_conv(features, features)

    def forward(self, x):
        x = self.inc(x)
        x1 = self.conv(x)
        x = x1 + x
        x1 = self.sconv(x)
        x = x1 + x
        x1 = self.conv(x)
        x = x1 + x
        x1 = self.sconv(x)
        x = x1 + x
        x1 = self.conv(x)
        x = x1 + x
        x = self.outc(x)
        return x


class LConvNetv2(nn.Module):
    def __init__(self, n_channels, n_classes, features, lkszie, lkblocknum):
        super(LConvNetv2, self).__init__()
        self.inc = double_conv(n_channels, features)
        self.LCblock = RepLKNetStage(channels=features, num_blocks=lkblocknum - 1, stage_lk_size=lkszie, drop_path=0,
                                     small_kernel=5, dw_ratio=1, ffn_ratio=4)
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=n_classes, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.convout = ReparamLargeKernelConv(in_channels=features, out_channels=n_classes, kernel_size=lkszie,
                                              stride=1, groups=1, small_kernel=5)

    def forward(self, x):
        x = self.inc(x)
        x = self.LCblock(x)
        x = self.convout(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, features)
        self.down11 = down(features, features * 2)
        self.down21 = down(features * 2, features * 4)
        self.down31 = down(features * 4, features * 8)
        self.down41 = down(features * 8, features * 16)
        self.up11 = up(features * 16, features * 8)
        self.up21 = up(features * 8, features * 4)
        self.up31 = up(features * 4, features * 2)
        self.up41 = up(features * 2, features)
        self.unet_1st_out = outconv(features, n_channels)

        self.inc0 = inconv(n_channels * 2, features)
        self.down12 = down(features, features * 2)
        self.down22 = down(features * 2, features * 4)
        self.down32 = down(features * 4, features * 8)
        self.down42 = down(features * 8, features * 16)
        self.up12 = up(features * 16, features * 8)
        self.up22 = up(features * 8, features * 4)
        self.up32 = up(features * 4, features * 2)
        self.up42 = up(features * 2, features)
        self.unet_2nd_out = outconv(features, n_classes)
        self.outc5 = outconv(features, n_classes)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)
        x = self.up11(x5, x4)
        x = self.up21(x, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)
        x1 = self.inc0(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x4 = self.down32(x3)
        x5 = self.down42(x4)
        x = self.up12(x5, x4)
        x = self.up22(x, x3)
        x = self.up32(x, x2)
        x = self.up42(x, x1)
        x = self.unet_2nd_out(x)
        return x


class UNetS_small(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNetS_small, self).__init__()
        self.inc = inconv(n_channels, features)
        self.down11 = down(features, features * 2)
        self.down21 = down(features * 2, features * 4)
        self.down31 = down(features * 4, features * 8)
        self.down41 = down(features * 8, features * 16)
        self.up11 = up(features * 16, features * 8)
        self.up21 = up(features * 8, features * 4)
        self.up31 = up(features * 4, features * 2)
        self.up41 = up(features * 2, features)
        self.unet_1st_out = outconv(features, n_classes)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)
        x = self.up11(x5, x4)
        x = self.up21(x, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        return x


class UNetsmall(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNetsmall, self).__init__()
        self.inc = inconv(n_channels, features)
        self.down11 = down(features, features * 2)
        self.down21 = down(features * 2, features * 4)
        self.down31 = down(features * 4, features * 8)
        self.up21 = up(features * 8, features * 4)
        self.up31 = up(features * 4, features * 2)
        self.up41 = up(features * 2, features)
        self.unet_1st_out = outconv(features, n_channels)

        self.inc0 = inconv(n_channels * 2, features)
        self.down12 = down(features, features * 2)
        self.down22 = down(features * 2, features * 4)
        self.down32 = down(features * 4, features * 8)
        self.up22 = up(features * 8, features * 4)
        self.up32 = up(features * 4, features * 2)
        self.up42 = up(features * 2, features)
        self.unet_2nd_out = outconv(features, n_classes)
        self.outc5 = outconv(features, n_classes)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x = self.up21(x4, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)
        x1 = self.inc0(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x4 = self.down32(x3)
        x = self.up22(x4, x3)
        x = self.up32(x, x2)
        x = self.up42(x, x1)
        x = self.unet_2nd_out(x)
        return x


class UNetsmall2(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNetsmall2, self).__init__()
        self.inc = inconv(n_channels, features)
        self.down11 = down(features, features * 2)
        self.down21 = down(features * 2, features * 4)
        self.down31 = down(features * 4, features * 8)
        self.up21 = up(features * 8, features * 4)
        self.up31 = up(features * 4, features * 2)
        self.up41 = up(features * 2, features)
        self.unet_1st_out = outconv(features, n_channels)

        self.inc0 = inconv(n_channels * 2, features)
        self.down12 = down(features, features * 2)
        self.down22 = down(features * 2, features * 4)
        self.down32 = down(features * 4, features * 8)
        self.up22 = up(features * 8, features * 4)
        self.up32 = up(features * 4, features * 2)
        self.up42 = up(features * 2, features)
        self.unet_2nd_out = outconv(features, n_classes)
        self.outc5 = outconv(features, n_classes)

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x = self.up31(x3, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)
        x1 = self.inc0(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x = self.up32(x3, x2)
        x = self.up42(x, x1)
        x = self.unet_2nd_out(x)
        return x


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size, sigma, high_pass=False):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=False)
        self.high_pass = high_pass
        self.weight = self.create_gaussian_kernel(kernel_size, sigma, high_pass)

    def create_gaussian_kernel(self, kernel_size, sigma, high_pass=False):
        x_cord = torch.arange(kernel_size) - kernel_size // 2
        y_cord = torch.arange(kernel_size) - kernel_size // 2
        x_grid, y_grid = torch.meshgrid(x_cord, y_cord)
        gauss = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2.0 * sigma ** 2))
        gauss /= gauss.sum()


        if high_pass:
            uniform_filter = torch.ones_like(gauss) / gauss.numel()
            gauss = uniform_filter - gauss
        return gauss.view(1, 1, kernel_size, kernel_size)

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        self.weight = self.weight.to(x.device)
        if x.dtype == torch.float16:
            self.weight = self.weight.to(x.device).to(torch.float16)
        elif x.dtype == torch.float32:
            self.weight = self.weight.to(x.device).to(torch.float32)
        return F.conv2d(x, self.weight, padding=padding)


class LargePNet(nn.Module):
    def __init__(self, n_channels, n_classes, upscale, lksize, lkblocknum):
        super(LargePNet, self).__init__()
        featnum = 4
        self.inc = inconvNB(n_channels, 16)
        self.SEIncept = SELayer(channel=16)
        self.conv1 = double_conv(in_ch=16, out_ch=n_channels)
        self.conv2 = double_conv(in_ch=2 * n_channels, out_ch=2)
        self.down = nn.AvgPool2d(3, 2, 1)
        self.upscale = upscale
        self.Unet512 = LConvNetv2(2 * n_channels, 4 ** (upscale - 1) *  n_classes, featnum * 16, lksize, lkblocknum)
        self.Unet256 = UNet(n_channels=2 * n_channels, n_classes= n_classes, features=featnum * 8)
        self.Unet128 = UNet(n_channels=2 * n_channels, n_classes= n_classes, features=featnum * 4)
        self.Unet64 = UNetsmall(n_channels=2 * n_channels, n_classes= n_classes, features=featnum * 2)
        self.Unet32 = UNetsmall2(n_channels=2 * n_channels, n_classes= n_classes, features=featnum)

        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.outconv = outconv(7, n_classes)
        self.sigmoid = nn.Sigmoid()

        self.yfilter1 = GaussianFilter(3, sigma=1.0, high_pass=True)  # 高通滤波器
        self.yfilter2 = GaussianFilter(3, sigma=1.0, high_pass=True)  # 高通滤波器
        self.y512_filter = GaussianFilter(3, sigma=1.0, high_pass=True)  # 高通滤波器
        self.y256_filter = GaussianFilter(3, sigma=2.0, high_pass=False)  # 低通滤波器
        self.y128_filter = GaussianFilter(3, sigma=2.0, high_pass=False)  # 低通滤波器
        self.y64_filter = GaussianFilter(3, sigma=2.0, high_pass=False)  # 低通滤波器
        self.y32_filter = GaussianFilter(3, sigma=2.0, high_pass=False)  # 低通滤波器

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.SEIncept(x1)
        x3 = self.conv1(x2)
        y = torch.cat([x, x3], dim=1)
        y512 = y
        y256 = self.down(y512)
        y128 = self.down(y256)
        y64 = self.down(y128)
        y32 = self.down(y64)
        # Go through Unet
        y512 = self.Unet512(y512)
        y256 = self.Unet256(y256)
        y128 = self.Unet128(y128)
        y64 = self.Unet64(y64)
        y32 = self.Unet32(y32)
        y = self.conv2(y)
        # Up-sample
        # print(y256.size())
        y512_out = self.pixel_shuffle(y512)
        y256_out = F.interpolate(y256, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale),
                                 mode='bilinear', align_corners=True)
        y128_out = F.interpolate(y128, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale),
                                 mode='bilinear', align_corners=True)
        y64_out = F.interpolate(y64, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale), mode='bilinear',
                                align_corners=True)
        y32_out = F.interpolate(y32, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale), mode='bilinear',
                                align_corners=True)
        y_out = F.interpolate(y, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale), mode='bilinear',
                              align_corners=True)

        # y512_out = self.y512_filter(y512_out)
        # y256_out = self.y256_filter(y256_out)
        # y128_out = self.y128_filter(y128_out)
        # y64_out = self.y64_filter(y64_out)
        # y32_out = self.y32_filter(y32_out)
        # y_out1 = self.yfilter1(y_out[:, 0:1, :, :])
        # y_out2 = self.yfilter2(y_out[:, 0:1, :, :])

        concat = y512_out + y256_out + y128_out + y64_out + y32_out
        out = self.sigmoid(concat)
        return out
