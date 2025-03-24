# 计算一维的高斯分布向量
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import cv2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(device)
    window=window.to(device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        img2=img2.to(device)
        img1 = img1.to(device)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class SLoss(nn.Module):
    def __init__(self, SLoss_weight=0.000):
        super(SLoss, self).__init__()
        self.SLoss_weight = SLoss_weight

    def forward(self,x):
        v=torch.sum(torch.abs(x))
        return self.SLoss_weight*v

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x,y,z):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w)+torch.mean(torch.abs(y-z))

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class HLoss(nn.Module):
    def __init__(self, HLoss_weight=0.5):
        super(HLoss, self).__init__()
        self.HLoss_weight = HLoss_weight

    def forward(self, x,y,z):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_diff = torch.pow((x[:, :, :h_x-2, :w_x-1] +x[:, :, 2: , :w_x-1]-2*x[:,:,1:h_x-1,:w_x-1]), 2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_diff = torch.pow((x[:, :, :h_x-1, :w_x-2] +x[:, :, :h_x-1 , 2:]-2*x[:,:,:h_x-1,1:w_x-1]), 2).sum()
        wh_diff=torch.pow((x[:, :, 1:, :w_x-1] -x[:, :, :h_x-1 , 1:]-x[:,:,1:,1:]+x[:,:,h_x-1,:w_x-1]), 2).sum()
        return self.HLoss_weight * 2 * (h_diff /count_h+ w_diff / count_w+2*wh_diff/count_w)+torch.mean(torch.abs(y-z))

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

