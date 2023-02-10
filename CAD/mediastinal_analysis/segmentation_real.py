import SimpleITK as sitk
from sys import argv
from os import path
import cv2
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes
from skimage.filters import roberts
import numpy as np
from skimage import transform as sktsf
from skimage import measure

DEBUG = True

def conventional_lung_cropping(CT):
    CT = np.transpose(CT, (1, 2, 0))
    _, _, d = CT.shape
    lung_seg = []
    for i in range(d):
        slice = CT[:, :, i].astype(np.uint8)
        _, binary_slice = cv2.threshold(slice, 0, 255, cv2.THRESH_BINARY)

        # floodfill
        im_floodfill = binary_slice.copy()
        h, w = binary_slice.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = binary_slice | im_floodfill_inv

        # erode
        kernel = np.ones((25, 25), np.uint8)
        im_out = cv2.erode(im_out, kernel, iterations=1)

        binary_slice = 255 - (binary_slice + (255 - im_out))
        _, binary_slice = cv2.threshold(binary_slice, 200, 255, cv2.THRESH_BINARY)
        
        # drop small connected areas
        labeled_img, num = measure.label(binary_slice, background=0, return_num=True)
        
        if num >= 3:
            dict = {}
            props = measure.regionprops(labeled_img)
            for idx in range(num):
                dict[idx] = props[idx].bbox_area
            sorted_dict = sorted(dict.items(), key=lambda x:x[1], reverse=True)
            max_idx, second_idx = sorted_dict[0][0] + 1, sorted_dict[1][0] + 1
            binary_slice = np.array((labeled_img == max_idx) | (labeled_img == second_idx)).astype(np.uint8) * 255

        lung_seg.append(binary_slice)

    # compute overlap along z axis, start from middle slice
    current_slice = lung_seg[d//2]
    for index in range(d//2, -1, -1):
        slice_to_check = lung_seg[index]
        labeled_img, num = measure.label(slice_to_check, background=0, return_num=True)
        blank_slice = np.zeros_like(slice_to_check)
        for i in range(1, num+1):
            current_connected_area = np.array(labeled_img == i).astype(np.uint8)
            if np.sum(current_connected_area * current_slice) > 0:
                blank_slice += current_connected_area * 255
        lung_seg[index] = blank_slice
        current_slice = lung_seg[index]

    current_slice = lung_seg[d//2]
    for index in range(d//2, d):
        slice_to_check = lung_seg[index]
        labeled_img, num = measure.label(slice_to_check, background=0, return_num=True)
        blank_slice = np.zeros_like(slice_to_check)
        for i in range(1, num+1):
            current_connected_area = np.array(labeled_img == i).astype(np.uint8)
            if np.sum(current_connected_area * current_slice) > 0:
                blank_slice += current_connected_area * 255
        lung_seg[index] = blank_slice
        current_slice = lung_seg[index]

    return lung_seg

def wc_transform(img, wc=None, ww=None):
    if wc != None:
        wmin = (wc*2 - ww) // 2
        wmax = (wc*2 + ww) // 2
    else:
        wmin = img.min()
        wmax = img.max()
    dfactor = 1.0 / (wmax - wmin)
    img = np.where(img < wmin, wmin, img)
    img = np.where(img > wmax, wmax, img)
    img = img.astype(np.float32)
    img = (img - wmin) * dfactor

    return img

def imdebug(name, img):
  if not DEBUG: return
  cv2.imshow(name, img)
  cv2.waitKey()

filename = argv[1]

exit(0)
#img = sitk.ReadImage('../data/test.nrrd')

#spacing = np.array(img.GetSpacing())
#origin = np.array(img.GetOrigin())
spacing = [1, 1, 1]

data = np.load('../data/test.npz')['CT']
data = np.transpose(data, (2, 0, 1)) # d, h, w

heat_volume = np.load('../data/test.npz')['mask']
heat_volume = heat_volume.astype(np.float32)
    
lung_seg = np.array(conventional_lung_cropping(data))
mediastinal_mask = []
for idx, lung_mask in enumerate(lung_seg):
  new_mask = np.zeros(lung_mask.shape)
  for h in range(lung_mask.shape[0]):
    for w in range(lung_mask.shape[1]):
        if lung_mask[h, :w].any() and lung_mask[h, w:].any():
            new_mask[h, w] = 1
  ''' for test1
  if heat_volume[idx].any():
    for h in range(lung_mask.shape[0]):
        if heat_volume[idx][h].any():
            new_mask[:h, :] = 0
            break
  else:
    new_mask[:20,:] = 0
  '''
  new_mask[:20,:] = 0
  mediastinal_mask.append(new_mask)
mediastinal_mask = np.array(mediastinal_mask)

lung_volume = data / 255 * mediastinal_mask
#lung_volume += lung_seg / 255 * 0.4
lung_volume = lung_volume.astype(np.float32)

#heat_volume = np.zeros(lung_volume.shape, dtype='float32')

heat_img = sitk.GetImageFromArray(heat_volume)
#heat_img.SetSpacing([1,1,0.5]) # for test1
#heat_img.SetSpacing([1,1,1.2]) # for test2

lung_img = sitk.GetImageFromArray(lung_volume)

#lung_img.SetSpacing([1,1,0.5]) # for test1
#lung_img.SetSpacing([1,1,1.2]) # for test2

sitk.WriteImage(lung_img, '../data/test.nrrd')
sitk.WriteImage(heat_img, '../data/test.heat.nrrd')
