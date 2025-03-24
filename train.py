import torch
import cv2
import numpy as np
import torchvision
from models import ResUnet, Dataset,Transformer
import torch.optim as optim
import torch.nn as nn
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
import numpy as np

from lossfunction import HLoss

def torch_normalize(x):
    x=(x-torch.min(x))/(torch.max(x)-torch.min(x))
    return x

def normalize(image):
    max = np.max(image)
    min = np.min(image)
    nor_image = (image - min) / (max - min)
    return nor_image


# 检测电脑上是否安装了对应版本的cuda
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
print('设备名称： ', device)
print('查看cuda版本： ', torch.version.cuda)

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='dataset/10f/', type=str)
parser.add_argument('--load_weight', default=False, type=bool)
args = parser.parse_args()

# 加载训练集和测试集
dataset=Dataset(args.data_folder, 128)
load=args.load_weight
trainloader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=True, pin_memory=True,drop_last=True)
psf_path='psf-2.tif'
Net = Transformer(
    img_dim=256,
    in_channel=2,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128*4,
    window_size=11,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[64, 128, 256, 512],
    input_dropout_rate=0
)

optimizer = optim.Adam(Net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

psf=np.array(Image.open(psf_path),dtype='float32')
psf=normalize(psf)
psf=transforms.ToTensor()(psf)
psf=psf.unsqueeze(0)
psf.unsqueeze(0)
psf=psf.to(device)

if load:
    Net.load_state_dict(
        torch.load('weights//TRUS_weights.pth'))
Net.to(device)

epochs = 500  # 训练两个epoch，batch_size = 4 (batch_size的大小定义在第一步torch.utils.data.DataLoader中)
e1 = cv2.getTickCount()  # 记录训练时间
addition = HLoss()

for epoch in range(epochs):
    total_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 得到inputs
        # print(i)

        norm_uaf_wf,norm_uaf,norm_gt_uaf = data
        # print(inputs.size())
        norm_uaf_wf,norm_uaf,norm_gt_uaf =  norm_uaf_wf.to(device),norm_uaf.to(device),norm_gt_uaf.to(device)
        gt=torch.cat([norm_gt_uaf],dim=1)

        optimizer1.zero_grad()
        outputs1 = Net(torch.cat([norm_uaf,norm_uaf_wf],dim=1))
        # outputs1=Net1(norm_uaf,norm_uaf_wf_2)
        outputs1=outputs1.to(device)
        out_psf=torch_normalize(torch.nn.functional.conv2d(outputs1, psf, bias=None,padding=12, dilation=1, groups=1))
        out_psf=out_psf.to(device)
        z = addition(outputs1,out_psf,norm_gt_uaf)
        loss=z
        loss.backward()
        optimizer.step()
        # total_loss1 += loss1.item()
        total_loss += loss.item()


        if (i + 1) % 100 == 0:  # 每1000次(就是4000张图像)输出一次loss
            print('第{}个epoch：第{:5d}次：目前3000f的训练损失loss为：{:.6f}'.format(epoch + 1, i + 1, total_loss/ 1000))
            total_loss = 0.0
            PATH1 = 'weights//TRUS_weight.pth'
            torch.save(Net1.state_dict(),PATH1)
