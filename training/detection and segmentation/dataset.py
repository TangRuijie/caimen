from __future__ import  absolute_import
from __future__ import  division
import numpy as np
import os
import cv2
from skimage import transform as sktsf
import random
import json
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

class Dataset:
    def __init__(self, filenames, base_path, is_train=True, direction="axial", use_crop=False):
        self.filenames = filenames
        self.is_train = is_train
        self.base_path = base_path
        self.direction = direction
        self.use_crop = use_crop

    def __getitem__(self, index):
        dir = self.filenames[index].strip()
        id = dir.split("/")[-1][:-4]

        CT = np.load(os.path.join(self.base_path, dir))
        mask = np.load(os.path.join(self.base_path, dir).replace('/CT/', '/seg_mask/'))
        
        do_vertical_fip = random.random() < 0.5
        do_horizontal_flip = random.random() < 0.5
        
        if self.is_train and do_vertical_fip:
            CT = CT[::-1,:]
            mask = mask[::-1,:]
        if self.is_train and do_horizontal_flip:
            CT = CT[:,::-1]
            mask = mask[:,::-1]

        CT = CT.astype(np.float32)
        CT = self.wc_transform(CT, 0, 400)
        CT = np.expand_dims(CT, 0)
        if self.direction == "axial":
            CT = sktsf.resize(CT, (1, 256, 256), anti_aliasing=True)
        else:
            CT = sktsf.resize(CT, (1, 256, 256), anti_aliasing=True)

        mask = np.where(mask > 1, 1, mask)
        mask = mask.astype(np.float32)
        mask = sktsf.resize(mask, (256, 256), anti_aliasing=True)
        mask = np.where(mask < 0.5, 0, 1)
        mask = np.expand_dims(mask, 0)
            
        return id, CT.astype(np.float32), mask.astype(np.float32)

    def __len__(self):
        return len(self.filenames)

    def wc_transform(self, img, wc=None, ww=None):
        if wc != None:
            wmin = (wc*2 - ww) // 2
            wmax = (wc*2 + ww) // 2
        else:
            wmin = img.min()
            wmax = img.max()
        dfactor = 1.0 / (wmax - wmin)
        img = np.where(img < wmin, wmin, img)
        img = np.where(img > wmax, wmax, img)
        img = (img - wmin) * dfactor

        return img