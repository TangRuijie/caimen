from __future__ import  absolute_import
from __future__ import  division
import numpy as np
import os
import cv2
from skimage import transform as sktsf
import random

class Dataset:
    def __init__(self, filenames, is_train=True, use_crop=False):
        self.filenames = filenames
        self.is_train = is_train
        self.use_crop = use_crop

    def __getitem__(self, index):
        info = self.filenames[index].strip()
        if self.use_crop:
            path, others = info.split(".npz ")
            path = path + ".npz"
            z1, z2, y1, y2, x1, x2, cls_label = others.split()
            z1, z2, y1, y2, x1, x2 = eval(z1), eval(z2), eval(y1), eval(y2), eval(x1), eval(x2)
        else:
            path, cls_label = info.split()
        cls_label = cls_label.strip()

        if path.find('/GPUFS') == -1:
            path = '/GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA/segmentation/' + path
        meta = np.load(path, allow_pickle=True)
        try:
            CT = meta['data'] # h, w, d
            seg_mask = meta['labels'] # d, h, w
        except:
            CT = meta['CT']
            seg_mask = meta['mask']
        if self.use_crop:
            CT = CT[y1:y2, x1:x2, z1:z2]
            seg_mask = seg_mask[z1:z2, y1:y2, x1:x2]
        if seg_mask.max() >= 3:
            seg_mask = np.where(seg_mask >= 3, 1, 0)
        else:
            seg_mask = np.where(seg_mask >= 1, 1, 0)
        seg_mask = seg_mask.astype(np.float32)
        d, _, _ = seg_mask.shape
        seg_mask = sktsf.resize(seg_mask, (d, 256, 256), anti_aliasing=True)
        seg_mask = np.where(seg_mask < 0.5, 0, 1)
        CT = CT.astype(np.float32)

        CT = self.wc_transform(CT, 0, 400)
        _, _, d = CT.shape
        CT = sktsf.resize(CT, (256, 256, d), anti_aliasing=True)

        CT = np.transpose(CT, (2, 0, 1)) # d, h, w
        
        do_vertical_fip = random.random() < 0.5
        do_horizontal_flip = random.random() < 0.5
        
        assert CT.shape == seg_mask.shape
        if self.is_train and do_vertical_fip:
            CT = CT[:, ::-1, :]
            seg_mask = seg_mask[:, ::-1, :]
        if self.is_train and do_horizontal_flip:
            CT = CT[:, :, ::-1]
            seg_mask = seg_mask[:, :, ::-1]
        
        return cls_label, CT.copy(), seg_mask.copy()

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