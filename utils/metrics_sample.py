import os
import glob
from PIL import Image

import numpy as np
import urllib.request
import scipy.linalg

from tqdm import tqdm_notebook as tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transforms = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


# class GeneratorDataset(torch.utils.data.Dataset):
#     def __init__(self, model, mex_len=5000, transform=None):
#         self.model = model
#         self.mex_len = mex_len
#         self.transforms = transform
#
#     def __len__(self):
#         return self.mex_len
#
#     def __getitem__(self, i):
#         path = self.files[i]
#         img = Image.open(path).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img


source_fid_mu = None
source_fid_sigma = None


# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
def calc_fid50k_full(config, network_url, source_loader: DataLoader, eval_loader: DataLoader, device):
    global source_fid_mu
    global source_fid_sigma

    if os.path.exists("features.pt") is False:
        print("checkpoint is not found, start download")
        urllib.request.urlretrieve(network_url, "features.pt")
    model = torch.jit.load("features.pt").eval().to(device)

    if source_fid_mu is None or source_fid_sigma is None:
        print(f"start compute source fid, dataloader length is {len(source_loader)}")
        source_mu, source_sigma = compute_fid(config, model, source_loader, device)
        source_fid_mu, source_fid_sigma = source_mu, source_sigma
    else:
        print(f"use cache source fid")
        source_mu, source_sigma = source_fid_mu, source_fid_sigma

    print(f"start compute eval fid, dataloader length is {len(eval_loader)}")
    eval_mu, eval_sigma = compute_fid(config, model, eval_loader, device)

    m = np.square(eval_mu - source_mu).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(eval_sigma, source_sigma), disp=False)
    fid = np.real(m + np.trace(eval_sigma + source_sigma - s * 2))
    return float(fid)


def compute_fid(config, model, dataloader: DataLoader, device):
    length = 0
    raw_mean = None
    raw_cov = None
    for step, batch in enumerate(tqdm(dataloader, "compute_fid")):
        images = batch
        if len(images.shape) == 2:
            images = images[0, :]
        feature = model(images.to(device)).cpu().numpy()

        length += feature.shape[0]
        x64 = feature.astype(np.float64)

        if raw_mean is None:
            raw_mean = np.zeros([feature.shape[1]], dtype=np.float64)
        raw_mean += x64.sum(axis=0)

        if raw_cov is None:
            raw_cov = np.zeros([feature.shape[1], feature.shape[1]], dtype=np.float64)
        raw_cov += x64.T @ x64

    mean = raw_mean / length
    cov = raw_cov / length
    cov = cov - np.outer(mean, mean)
    return mean, cov


def test():
    dataset = ImagePathDataset(glob.glob(f"/content/images/**/*.jpg", recursive=True)[:2], transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=True, pin_memory=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"
    calc_fid50k_full({}, network_url, dataloader, dataloader, device)
