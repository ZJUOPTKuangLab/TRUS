from torch.utils import data
from pathlib import Path
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import torch

def cal_ratio(Net, vs, uaf,val,image_size,fil):
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((int(image_size), int(image_size)),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    mask = uaf.copy()
    mask[mask < val] = 0
    mask[mask >= val] = 1
    if fil:
        vs = transform(Image.fromarray(vs*15/65535.0))
        uaf = transform(Image.fromarray(uaf*15/65535.0))
        vs=vs.unsqueeze(0)
        vs.unsqueeze(0)
        uaf=uaf.unsqueeze(0)
        uaf.unsqueeze(0)
        vs=vs.to(device)
        uaf=uaf.to(device)
        vs_filter = Net(vs)
        vs=np.squeeze(vs_filter['out'].cpu().detach().numpy())
        uaf_filter = Net(uaf)
        uaf=np.squeeze(uaf_filter['out'].cpu().detach().numpy())
    r = vs.copy() / (uaf.copy() + 1e-34)
    ratio = r * mask
    ratio[ratio > 3] = 0
    return ratio / 3

def normalize(image):
    max = np.max(image)
    min = np.min(image)
    nor_image = (image - min) / (max - min)
    return nor_image


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png','tif']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # self.paths = [p for ext in exts for p in cc.glob(f'**/')]
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        # self.paths = [sub_p for sub_p in Path(f'{folder}').iterdir() if sub_p.is_dir()]
        self.transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        self.transform1 =  transforms.Compose([
            transforms.Resize((int(image_size*0.25), int(image_size*0.25))),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),

            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        self.transform2 =  transforms.Compose([

            transforms.Resize((int(image_size*2), int(image_size*2)),interpolation=Image.BICUBIC),
            # transforms.CenterCrop(image_size*0.5),
            transforms.ToTensor(),

            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        data_path='10f'
        wf_path='wf'
        gt_path='gt'

        path=str(path)
        path=path.replace('\\','/')
        uaf_path=path
        wfuaf_path =path.replace(data_path,wf_path)
        gt_uaf_p =path.replace(data_path,gt_path)

        uaf_wf = np.array(Image.open(wfuaf_path),dtype='float32')
        uaf = np.array(Image.open(uaf_path),dtype='float32')
        gt_uaf = np.array(Image.open(gt_uaf_p),dtype='float32')

        norm_uaf_wf=Image.fromarray(normalize(uaf_wf.copy()))
        norm_uaf=Image.fromarray(normalize(uaf.copy()))

        norm_gt_uaf=Image.fromarray(normalize(gt_uaf.copy()))

        return self.transform(norm_uaf_wf), self.transform(norm_uaf),self.transform2(norm_gt_uaf)

