import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn

class Discriminator(nn.Module):
    # Conventional model
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels

        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))


        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    def forward(self, img):
        return self.sigmoid(self.model(img))

class DiscriminatorS(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorS, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 3), int(in_width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class MultiScaleDiscriminator(nn.Module):
    # LargeP-GAN
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 3), int(in_width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)
        self.D1 = DiscriminatorS((1, 512, 512))
        self.D2 = DiscriminatorS((1, 256, 256))
        self.D3 = DiscriminatorS((1, 128, 128))
        self.D4 = DiscriminatorS((1, 64, 64))
        self.down = nn.AvgPool2d(3, 2, 1)
        self.upscale = 1
    def forward(self, x):
        y512 = x
        y256 = self.down(y512)
        y128 = self.down(y256)
        y64 = self.down(y128)
               
        y512 = self.D1(y512)
        y256 = self.D2(y256)
        y128 = self.D3(y128)
        y64 = self.D4(y64)
       
        
        y512_out = y512
        y256_out = F.interpolate(y256, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale),
                                 mode='bilinear', align_corners=True)
        y128_out = F.interpolate(y128, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale),
                                 mode='bilinear', align_corners=True)
        y64_out = F.interpolate(y64, size=(y512.shape[2] * self.upscale, y512.shape[3] * self.upscale), mode='bilinear',
                                align_corners=True)

        total =y512_out + y256_out + y128_out + y64_out
        
        return total
