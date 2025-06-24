# Basic components
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast
        
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                              dtype=None),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                              dtype=None),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class LK_conv(nn.Module):
    def __init__(self, in_ch, out_ch, lksize):
        super(LK_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, lksize, lksize), padding=(1, lksize//2, lksize//2), dilation = 1, groups = 1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2),           
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, dilation = 1, groups = 1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2), 
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class RepLK_conv(nn.Module):
    def __init__(self, in_ch, out_ch, lksize):
        super(RepLK_conv, self).__init__()
        self.lkconv = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, dilation = 1, groups = 1),
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, lksize, lksize), padding=(3//2, lksize//2, lksize//2), dilation = 1, groups = out_ch),
            nn.LeakyReLU(0.2),#
            nn.InstanceNorm3d(out_ch),
        )
        self.skconv = nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=3//2, dilation = 1, groups = out_ch),
            nn.InstanceNorm3d(out_ch),
        )
        self.Nonlinear = nn.Sequential(
            nn.LeakyReLU(0.2),
        )
        self.Pointwise = nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, padding=0, dilation = 1, groups = 1),
        )
        self.convffn = nn.Sequential(
            nn.InstanceNorm3d(out_ch),
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch*4, kernel_size=1, padding=0, dilation = 1, groups = 1),
            nn.GELU(),
            nn.Conv3d(in_channels=out_ch*4, out_channels=out_ch, kernel_size=1, padding=0, dilation = 1, groups = 1),
            nn.LeakyReLU(0.2),#
        ) 
            
    def forward(self, x):
        x = self.lkconv(x)
        x1 = self.skconv(x)
        x = x + x1
        x = self.Nonlinear(x)
        x1 = self.Pointwise(x)
        x = x + x1
        x1 = self.convffn(x)
        x = x + x1
        return x
        
class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.InstanceNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
class double_convNB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_convNB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        super(Upsampler, self).__init__(*m)


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
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
            # self.up = Upsampler(default_conv, 2, in_ch, act=False)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = self.conv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.Droup = nn.Dropout(p=0.1,inplace=False)

    def forward(self, x):
        x = self.conv(x)
        # x = self.Droup(x)
        return x

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

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
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
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
        
class UNetsmall(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNetsmall, self).__init__()
        featnum = features
        self.inc = inconv(n_channels, featnum*2)
        self.down1 = down(featnum*2, featnum*4)
        self.down2 = down(featnum*4, featnum*8)
        self.up1 = up(featnum*8, featnum*4)
        self.up2 = up(featnum*4, featnum*2)
        self.outc = outconv(featnum*2, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        #x = self.sigmoid(x)
        return x

class UNetsmall2(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(UNetsmall2, self).__init__()
        featnum = features
        self.inc = inconv(n_channels, featnum*2)
        self.down1 = down(featnum*2, featnum*4)
        self.up1 = up(featnum*4, featnum*2)
        self.outc = outconv(featnum*2, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        x = self.outc(x)
        #x = self.sigmoid(x)
        return x

class LConvNetv2(nn.Module):
    def __init__(self, n_channels, n_classes, features, lksize, lkblocknum):
        super(LConvNetv2, self).__init__()
        self.inc = double_conv2d(n_channels, features)
        self.LCblock = RepLKNetStage(channels = features, num_blocks = lkblocknum-1, stage_lk_size = lksize, drop_path = 0, small_kernel = 5, dw_ratio=1, ffn_ratio=4)
        self.conv1 = nn.Conv3d(in_channels=features, out_channels=n_classes, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.convout = ReparamLargeKernelConv(in_channels=features, out_channels=n_classes, kernel_size= lksize,stride=1, groups=1,small_kernel=5)
        
    def forward(self, x):
        x = self.inc(x)
        x = self.LCblock(x)
        x = self.convout(x)
        return x
        
class LConvNetv3(nn.Module):
    def __init__(self, n_channels, n_classes, features, lksize):
        super(LConvNetv3, self).__init__()
        self.inc = double_conv(n_channels, features)
        self.LCblock = LK_conv(features, features, lksize)
        self.convout = LK_conv(features, n_classes, lksize)
        
    def forward(self, x):
        x = self.inc(x)
        x1 = self.LCblock(x)
        x = x + x1
        x1 = self.LCblock(x)
        x = x + x1
        x1 = self.LCblock(x)
        x = x + x1
        x = self.convout(x)
        return x 

class LConvNetv3dw(nn.Module):
    def __init__(self, n_channels, n_classes, features, lksize):
        super(LConvNetv3dw, self).__init__()
        self.inc = double_conv(n_channels, features)
        self.LCblock = RepLK_conv(features, features, lksize)
        self.convout = RepLK_conv(features, n_classes, lksize)
        
    def forward(self, x):
        x = self.inc(x)
        x1 = self.LCblock(x)
        x = x + x1
        x1 = self.LCblock(x)
        x = x + x1
        x1 = self.LCblock(x)
        x = x + x1
        x = self.convout(x)
        return x 
        
class LConvNetv4(nn.Module):
    def __init__(self, n_channels, n_classes, features):
        super(LConvNetv3, self).__init__()
        self.inc = double_conv(n_channels, features)
        self.down = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up1 = nn.ConvTranspose3d(4*features, 2*features, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up2 = nn.ConvTranspose3d(2*features, features, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = double_conv(features, 2*features)
        self.conv2 = double_conv(2*features, 4*features)
        self.conv3 = double_conv(4*features, 2*features)
        self.conv4 = double_conv(2*features, features)
        self.LCblock25 = LK_conv(features, features, 25)
        self.LCblock13 = LK_conv(2*features, 2*features, 25)
        self.LCblock7 = LK_conv(4*features, 4*features, 25)
        self.convout = LK_conv(features, n_classes, 25)
        
    def forward(self, x):
        x = self.inc(x)   ## 1*32*6*256*256
        x1 = self.LCblock25(x) ## 1*32*6*256*256
        x2 = self.down(x1) # 1*32*6*128*128
        x2 = self.conv1(x2) #1*64*6*128*128
        x2 = self.LCblock13(x2) ## 1*64*6*128*128
        x3 = self.down(x2) #1*64*6*64*64
        x3 = self.conv2(x3) #1*128*6*64*64
        x3 = self.LCblock7(x3) #1*128*6*64*64
        x3 = self.up1(x3)  # 1*64*6*128*128
        x4 = torch.cat([x2,x3],dim=1) #1*128*6*128*128
        x4 = self.conv3(x4)  #1*64*6*128*128
        x4 = self.LCblock13(x4) #1*64*6*128*128
        x4 = self.up2(x4)  #1*32*6*256*256
        x5 = torch.cat([x1,x4],dim=1) #1*64*256*256
        x_out = self.conv4(x5) #1*32*256*256
        x_out = self.convout(x_out)    
        return x_out 
  
class GaussianFilter(nn.Module):  
    def __init__(self, kernel_size, sigma, high_pass=False):  
        super(GaussianFilter, self).__init__()  
        self.kernel_size = kernel_size  
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=True)  
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
        if x.dtype == torch.float16:  
            self.weight = self.weight.to(x.device).to(torch.float16)  
        elif x.dtype == torch.float32:  
            self.weight = self.weight.to(x.device).to(torch.float32) 
        return F.conv2d(x, self.weight, padding=padding) 
        

class LargePNet3D(nn.Module):
    def __init__(self, n_channels, n_classes,  lksize, lkblocknum):
        super(LargePNet3D, self).__init__()
        featnum = 4      
        self.inc = inconvNB(n_channels, 16)
        self.selayer = SELayer3D(16)
        self.conv = double_conv(16, 1)
        self.down2d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down3d = nn.MaxPool3d(2)
        
        self.Unet512 = LConvNetv3(1,1,featnum * 4, lksize)
        self.Unet256 = UNet(n_channels=n_channels, n_classes=1, features=featnum * 8)
        self.Unet128 = UNet(n_channels=n_channels, n_classes=1, features=featnum * 6)
        self.Unet64 = UNetsmall(n_channels=n_channels, n_classes=1, features=featnum * 4)
        self.Unet32 = UNetsmall2(n_channels=n_channels, n_classes=1, features=featnum * 2)
        
        self.upscale = 1
        self.outconv = outconv(6, n_classes)
        self.sigmoid = nn.Sigmoid()
        
        self.yfilter1 = GaussianFilter(7, sigma=1.0, high_pass=True)
        self.yfilter2 = GaussianFilter(7, sigma=1.0, high_pass=True)
        self.y512_filter = GaussianFilter(7, sigma=1.0, high_pass=True)
        self.y256_filter = GaussianFilter(7, sigma=2.0, high_pass=False)
        self.y128_filter = GaussianFilter(7, sigma=2.0, high_pass=False)
        self.y64_filter = GaussianFilter(7, sigma=2.0, high_pass=False)
        self.y32_filter = GaussianFilter(7, sigma=2.0, high_pass=False)

    def forward(self, y):
        y256 = self.down2d(y)
        y128 = self.down2d(y256)
        y64 = self.down2d(y128)
        y32 = self.down2d(y64)
        y512 = y
        y512 = self.Unet512(y512)
        y256 = self.Unet256(y256)
        y128 = self.Unet128(y128)
        y64 = self.Unet64(y64)
        y32 = self.Unet32(y32)

        target_depth, target_height, target_width = y.shape[2:]  
        upsampled_depth = target_depth * self.upscale  
        upsampled_height = target_height * self.upscale  
        upsampled_width = target_width * self.upscale         

        y32 = F.interpolate(y32, size=(upsampled_depth, upsampled_height, upsampled_width), mode='trilinear', align_corners=False) 
        y64 = F.interpolate(y64, size=(upsampled_depth, upsampled_height, upsampled_width), mode='trilinear', align_corners=False)  
        y128 = F.interpolate(y128, size=(upsampled_depth, upsampled_height, upsampled_width), mode='trilinear', align_corners=False)  
        y256 = F.interpolate(y256, size=(upsampled_depth, upsampled_height, upsampled_width), mode='trilinear', align_corners=False)  

        y = y.squeeze(1)
        y512 = y512.squeeze(1)
        y256 = y256.squeeze(1)
        y128 = y128.squeeze(1)
        y64 = y64.squeeze(1)
        y32 = y32.squeeze(1)
        y = y.permute(1, 0, 2, 3) 
        y512 = y512.permute(1, 0, 2, 3)
        y256 = y256.permute(1, 0, 2, 3)
        y128 = y128.permute(1, 0, 2, 3)
        y64 = y64.permute(1, 0, 2, 3)
        y32 = y32.permute(1, 0, 2, 3)
        y = self.yfilter1(y)
        # y512 = self.y512_filter(y512)
        # y256 = self.y256_filter(y256)
        # y128 = self.y128_filter(y128)
        # y64 = self.y64_filter(y64)
        # y32 = self.y32_filter(y32)
        out = y + y512 + y256 + y128 + y64 + y32
        out = out.permute(1, 0, 2, 3).unsqueeze(0) 
        out = self.sigmoid(out)    
        return out 
