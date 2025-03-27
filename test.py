import torch
import torch.nn as nn
from models import Unet, Unet2, Dataset, Transformer
from PIL import Image
import numpy as np
from torchvision import transforms, utils
import utils
from skimage.filters import rank
from skimage.morphology import disk, ball
import time
import os
def remove_outliers(image, radius=1, threshold=2):
    footprint_function = disk if image.ndim == 2 else ball
    footprint = footprint_function(radius=radius)
    median_filtered = rank.median(image, footprint)
    outliers = (
            (image > median_filtered + threshold)
            | (image < median_filtered - threshold)
    )
    output = np.where(outliers, median_filtered, image)
    return output

def normalize(image):
    max = np.max(image)
    min = np.min(image)
    nor_image = (image - min) / (max - min)
    return nor_image


device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
# device = torch.device('cpu')
print('设备名称： ', device)
print('查看cuda版本： ', torch.version.cuda)

# images = Image.fromarray(np.array(Image.open(path), dtype='float32') / 255.0)
Net = Transformer(
    img_dim=128,
    in_channel=2,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128 * 4,
    window_size=11,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[64, 128, 256, 512],
    input_dropout_rate=0

)

Net.load_state_dict(
    torch.load('weights//TRUS_weights.pth'))
# Net.to(device)
Net.eval()

image_size1 = 192
image_size2 = 192
batch=192
xs=image_size1/batch
transform = transforms.Compose([
    transforms.Resize((int(batch),int(batch)),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])

path1="dataset\\simulation\\"

for j in range(0,1):
    for k in range(0, 15):

        wfuaf_path = str(path1) +'wf\\wf_uaf_'+str(k+1)+'.tif'
        uaf_path = str(path1) +'sofi\\uaf_20_'+str(k+1)+'.tif'

        uaf_wf = np.array(Image.open(wfuaf_path), dtype='float32')
        uaf = np.array(Image.open(uaf_path), dtype='float32')
        save = np.zeros([image_size2 * 2, int(image_size1 * 2)])
        save = save[np.newaxis, np.newaxis, :, :]

        # if k== 0:
        norm_uaf = normalize(uaf)
        norm_uaf_wf = normalize(uaf_wf)
        # else:
        #     norm_uaf = normalize(vs)
        #     norm_uaf_wf = normalize(vs_wf)

        for i in range(0,int(xs*xs)):
            print(i)
        # ratio_20=cal_ratio(vs,uaf,10,512,False)
            norm_uaf1 =  norm_uaf[int(int(i / xs) * batch):int((int(i / xs) + 1) * batch), int((i % xs) * batch):int((i % xs + 1) * batch)]
            norm_uaf_wf1 =norm_uaf_wf[int(int(i /xs) * batch):int((int(i /xs) + 1) * batch), int((i % xs) * batch):int((i % xs + 1) * batch)]

            uaf1 = Image.fromarray(norm_uaf1)
            uaf_wf1 = Image.fromarray(norm_uaf_wf1)
            print(norm_uaf.shape)

            uaf1 = transform(uaf1).unsqueeze(0)
            uaf_wf1 = transform(uaf_wf1).unsqueeze(0)

            a=time.time()
            outputs = Net(torch.cat([uaf1, uaf1], dim=1))
            b=time.time()
            print(b-a)
            save_image = outputs
            print(save_image.shape)
            save[0, 0, int(int(i / xs) * batch*2):int((int(i / xs) + 1) *batch*2),int((i % xs) *batch*2):int((i % xs + 1) * batch*2)] = save_image.detach().numpy()

        ratio = Image.fromarray(np.squeeze(np.uint8(normalize(save) * 255.0)))
        ratio = ratio.resize((image_size2, image_size1))
        ratio = Image.fromarray(np.array(ratio))
        save_path=path1+"trus\\"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ratio.save(save_path+"trus_"+str(k+1)+".tif")

