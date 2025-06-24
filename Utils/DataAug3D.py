import numpy as np  
import random 
from PIL import Image 
import scipy.ndimage as ndimage

def random_cropXY(raw,gt,aimingsize):
    # raw: 512*512*6
    d1, d2 ,d3 = raw.shape
    D1, D2, D3 = gt.shape
    scaler = D1/d1
    x = np.random.randint(low=0, high= d1 - aimingsize)
    y = np.random.randint(low=0, high= d2 - aimingsize)
    crop_raw = np.random.randn(aimingsize, aimingsize, d3)
    crop_gt = np.random.randn(aimingsize, aimingsize, d3)
    for i in range(d3):
        crop_raw[...,i] = raw[x:x+aimingsize, y:y+aimingsize,i]
        crop_gt[...,i] = gt[x:x+aimingsize, y:y+aimingsize,i]
    return crop_raw, crop_gt

def random_cropXY_in(raw,gt,aimingsize,x,y):
    # raw: 512*512*6
    d1, d2 ,d3 = raw.shape
    D1, D2, D3 = gt.shape
    scaler = D1/d1
    crop_raw = np.random.randn(aimingsize, aimingsize, d3)
    crop_gt = np.random.randn(aimingsize, aimingsize, d3)
    for i in range(d3):
        crop_raw[...,i] = raw[x:x+aimingsize, y:y+aimingsize,i]
        crop_gt[...,i] = gt[x:x+aimingsize, y:y+aimingsize,i]
    return crop_raw, crop_gt
    
def random_rotate(raw, gt ,state):
    values = state 
    d1, d2 ,d3 = raw.shape
    if values <3:
        out_raw = raw
        out_gt = gt
    if values == 3:
        angle = np.random.uniform(0, 360)
        out_raw = np.random.randn(d1, d2, d3)
        out_gt = np.random.randn(d1, d2, d3)  
        for i in range(d3):
            raw_slice = ndimage.rotate(raw[...,i],angle,reshape=False)
            gt_slice = ndimage.rotate(gt[...,i],angle,reshape=False) 
            out_raw[...,i] = raw_slice
            out_gt[...,i] = gt_slice
    return out_raw, out_gt
        
def random_cropXYZ(raw,gt,aimingsize, aimingsizez):
    # raw: 512*512*6
    d1, d2 ,d3 = raw.shape
    D1, D2, D3 = gt.shape
    scaler = D1/d1
    x = np.random.randint(low=0, high= d1 - aimingsize)
    y = np.random.randint(low=0, high= d2 - aimingsize)
    z = np.random.randint(low=0, high= d3 - aimingsizez)
    
    crop_raw = np.random.randn(aimingsize, aimingsize, aimingsizez)
    crop_gt = np.random.randn(aimingsize, aimingsize, aimingsizez)
    crop_raw = raw[x:x+aimingsize, y:y+aimingsize, z:z+aimingsizez]
    crop_gt = gt[x:x+aimingsize, y:y+aimingsize,z:z+aimingsizez]    
    return crop_raw, crop_gt

def random_cropXYZ_in(raw,gt,aimingsize, aimingsizez, x, y, z):
    # raw: 512*512*6
    d1, d2 ,d3 = raw.shape
    D1, D2, D3 = gt.shape
    scaler = D1/d1
    
    crop_raw = np.random.randn(aimingsize, aimingsize, aimingsizez)
    crop_gt = np.random.randn(aimingsize, aimingsize, aimingsizez)
    crop_raw = raw[x:x+aimingsize, y:y+aimingsize, z:z+aimingsizez]
    crop_gt = gt[x:x+aimingsize, y:y+aimingsize,z:z+aimingsizez]    
    return crop_raw, crop_gt
    
def flip_XY(raw,gt,state):
    if state==1:
        out_raw = raw
        out_gt = gt
    if state==2:
        out_raw = raw[:, ::-1, :]
        out_gt = gt[:, ::-1, :]
    if state == 3:
        out_raw = raw[::-1, :, :]
        out_gt = gt[::-1, :, :]  
    return out_raw, out_gt

def flip_Z(raw,gt,state):
    if state==1:
        out_raw = raw
        out_gt = gt
    if state==2:
        out_raw = raw[:, :, ::-1]
        out_gt = gt[:, :, ::-1]
    return out_raw, out_gt
    
def rotate_XY(raw,gt,state):
    if state==1:
        out_raw = raw
        out_gt = gt
    if state==2:
        out_raw = np.rot90(raw,1)
        out_gt = np.rot90(gt,1)
    if state == 3:
        out_raw = np.rot90(raw,-1)
        out_gt = np.rot90(gt,-1)
    return out_raw, out_gt

def DataAug(raw,gt,aimingsize):
    values = [1,2,3]
    random_numbers = np.random.choice(values, size=1)
    out_raw, out_gt = random_rotate(raw,gt,random_numbers)
    out_raw, out_gt = random_cropXY(out_raw,out_gt,aimingsize)
    values = [1, 2, 3]
    random_numbers = np.random.choice(values, size=1) 
    out_raw, out_gt = flip_XY(out_raw, out_gt, random_numbers)
    random_numbers = np.random.choice(values, size=1) 
    out_raw, out_gt = rotate_XY(out_raw, out_gt, random_numbers)

    values = [1, 2]
    random_numbers = np.random.choice(values, size=1) 
    out_raw, out_gt = flip_Z(out_raw, out_gt,random_numbers)
    return out_raw, out_gt

def DataAugZ(raw,gt,aimingsize,aimingsizez):
    # values = [1,2,3]
    # random_numbers = np.random.choice(values, size=1)
    # out_raw, out_gt = random_rotate(raw,gt,random_numbers)
    out_raw, out_gt = random_cropXYZ(raw,gt,aimingsize,aimingsizez)
    values = [1, 2, 3]
    random_numbers = np.random.choice(values, size=1) 
    out_raw, out_gt = flip_XY(out_raw, out_gt, random_numbers)
    random_numbers = np.random.choice(values, size=1) 
    out_raw, out_gt = rotate_XY(out_raw, out_gt, random_numbers)
    return out_raw, out_gt

def DataAugZ_in(raw,gt,aimingsize,aimingsizez, x, y, z, state1, state2):
    out_raw, out_gt = random_cropXYZ_in(raw,gt,aimingsize,aimingsizez, x, y, z)
    out_raw, out_gt = flip_XY(out_raw, out_gt, state1)
    out_raw, out_gt = rotate_XY(out_raw, out_gt, state2)
    return out_raw, out_gt