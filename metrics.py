from math import log10
import os

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim


def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse = diff.pow(2).mean()
    psnr = -10 * log10(mse)
    return float(psnr)


def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)


if __name__ == '__main__':
    gt = 'haadf_data_test/gt_enhance'
    path = os.listdir(gt)
    psnr, ssim1 = 0., 0.
    for name in path:
        in_path = os.path.join(gt, name)
        out_path = os.path.join('ours_result_enhance', name)
        in_img = torchvision.io.read_image(in_path).float()
        out_img = torchvision.io.read_image(out_path).float()
        in_img = torch.unsqueeze(in_img, 0)
        out_img = torch.unsqueeze(out_img, 0)
        psnr += calc_psnr(in_img, out_img)
        ssim1 += calc_ssim(in_img, out_img)
    print(psnr / len(path), ssim1 / len(path))
