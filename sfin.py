import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate
from ptflops import get_model_complexity_info as info


ch = 64
n_blocks = 8



class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        height, width = ffted.shape[-2:]
        coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
        coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
        ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        return output


class SpectralTransform(nn.Module):
    def __init__(self):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.fu = FourierUnit(ch // 2, ch // 2)
        self.conv2 = nn.Conv2d(ch, ch // 2, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        x = self.conv2(torch.cat([x, x2], dim=1))
        return x


class FFC(nn.Module):
    def __init__(self):
        super(FFC, self).__init__()
        self.convl2l = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.convl2g = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.convg2l = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        self.convg2g = SpectralTransform()

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class SFIB(nn.Module):
    def __init__(self):
        super(SFIB, self).__init__()
        self.ffc = FFC()
        self.bn_l = nn.BatchNorm2d(ch // 2)
        self.bn_g = nn.BatchNorm2d(ch // 2)
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class ResnetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SFIB()
        self.conv2 = SFIB()

    def forward(self, x):
        x_l, x_g = torch.split(x, (ch // 2, ch // 2), dim=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = torch.cat((x_l, x_g), dim=1)
        return out


class SFIN(nn.Module):
    def __init__(self):
        super(SFIN, self).__init__()
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(ResnetBlock())
        self.body = nn.Sequential(*self.blocks)
        self.head_conv = nn.Conv2d(1, ch, 3, 1, 1)
        self.tail_conv = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, x):
        x = self.head_conv(x)
        shortcut = x
        x = self.body(x)
        x += shortcut
        x = self.tail_conv(x)
        return x


if __name__ == '__main__':
    flops, params = info(SFIN(), (1, 256, 256), as_strings=False,
                         print_per_layer_stat=False, verbose=False)
    print(flops, params)
