# unet.py
#

from __future__ import division

from math import sqrt
import os

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from sfin import SFIN



if __name__ == '__main__':

    unet = nn.DataParallel(SFIN()).cuda()
    weights = torch.load("ours_enhance_tem_500.pth")
    unet.load_state_dict(weights['model_state_dict'])
    unet.eval()  
    torch.set_grad_enabled(False)

    for i in range(5):
        name = str(i) + '.png'
        in_path = os.path.join('haadf_data_test/noisy', name)
        out_path = os.path.join('ours_result_enhance', name)
        in_img = torchvision.io.read_image(in_path).cuda()
        if in_img.shape[0] == 3:
            in_img = in_img[:1]
        in_img = torch.unsqueeze(in_img, 0).float()
        out_img = unet(in_img)
        out_img = torch.clip_(out_img, 0, 255)
        out_img = torch.squeeze(out_img, 0).byte()
        torchvision.io.write_png(out_img.cpu(), out_path)
        print(i)
