import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, img3, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu3_sq = mu3.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_mu3 = mu1 * mu3
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma3_sq = F.conv2d(img3 * img3, window, padding=window_size // 2, groups=channel) - mu3_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma13 = F.conv2d(img1 * img3, window, padding=window_size // 2, groups=channel) - mu1_mu3

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    x2=torch.sqrt(sigma2_sq)
    x3=torch.sqrt(sigma3_sq)
    # ssim_map_12 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # ssim_map_13 = (2 * sigma13 + C2) / (sigma1_sq + sigma3_sq + C2)

    ssim_map_12 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map_13 = ((2 * mu1_mu3 + C1) * (2 * sigma13 + C2)) / ((mu1_sq+mu3_sq+C1)*(sigma1_sq + sigma3_sq + C2))

    if size_average:
        ssim_map = torch.where(mu2>mu3, ssim_map_12, ssim_map_13)
        # ssim_map = (ssim_map_13+ssim_map_12) / 2
        return ssim_map.mean()
    else:
        ssim_map = torch.where(mu2>mu3, torch.abs(ssim_map_12), torch.abs(ssim_map_13))
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, img3, window_size=16, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, img3,  window, window_size, channel, size_average)