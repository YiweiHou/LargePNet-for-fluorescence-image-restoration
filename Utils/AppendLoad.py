import glob
import tifffile
import numpy as np
from tqdm import tqdm
import torch


def AppendLoad(Path, Transtorch=True):
    all_imgs_path = glob.glob(Path)
    images = []

    for i, img_path in tqdm(enumerate(all_imgs_path), total=len(all_imgs_path)):
        img = tifffile.imread(img_path)
        img = np.array(img) / 65535
        images.append(img)
    images = np.array(images)
    if Transtorch == True:
        images = torch.tensor(images, dtype=torch.float16).unsqueeze(1)
    return images