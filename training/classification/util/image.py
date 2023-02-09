import os
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from .basic import mkdir

def norm2image(image,minp=None,maxp=None):
    if minp is None:
        if isinstance(image, np.ndarray):
            minp = np.min(image)

        if isinstance(image, torch.Tensor):
            minp = torch.min(image)

    if maxp is None:
        if isinstance(image, np.ndarray):
            maxp = np.max(image)

        if isinstance(image, torch.Tensor):
            maxp = torch.max(image)

    image = image - minp
    maxp = maxp - minp

    image = image * (255 / maxp)

    if isinstance(image, np.ndarray):
        image = image.astype('uint8')

    if isinstance(image, torch.Tensor):
        image = image.short()

    return image

def tensor2im(images, intype, outtype = 'NHWC', minp = None, maxp = None, norm_to_255 = True):
    """only process one image"""


    """"Converts a Tensor array into a numpy image array with N x H x W x C.
        When mode is 'gray', C = 1

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """


    if isinstance(images, torch.Tensor):  # get the data from a variable
        images = images.cpu().float().numpy()  # convert it into a numpy array

    image_shape = images.shape
    assert len(intype) == len(image_shape)

    intype = intype.upper()

    if 'N' not in intype:
        images = images[np.newaxis,:]
        intype = 'N' + intype

    if 'C' not in intype:
        images = images[np.newaxis,:]
        intype = 'C' + intype

    trans_tuple = [0] * 4
    for i, e in enumerate(intype):
        trans_tuple[outtype.index(e)] = i

    trans_tuple = tuple(trans_tuple)
    images = np.transpose(images, trans_tuple)

    C_index = outtype.lower().index('c')
    if images.shape[C_index] == 1:
        images = np.repeat(images, 3, axis = C_index)

    if norm_to_255:
        if isinstance(minp,type(None)):
            minp = np.min(images)

        if isinstance(maxp, type(None)):
            maxp = np.max(images)

        images = images - minp
        maxp = maxp - minp

        images = images * (255 / maxp)
        images = images.astype('uint8')
    return images

def cat_image(images,h_num = 10,factor = 0.75,in_type='HWC', out_type = 'HWC'):
    """

    Args:
        images: iamges with the same shape N x C x H x W or N x H x W x C
        h_num: images per line, -1 means all images in one line
        factor: the ratio factor for image size
        in_type: HWC or CHW
        out_type: HWC or CHW

    Returns:

    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()  # convert it into a numpy array

    if isinstance(images, list):
        for i, im in enumerate(images):
            if isinstance(im, torch.Tensor):
                images[i] = im.cpu().float().numpy()

        images = np.stack(images, axis= 0)

    if in_type == 'HWC':
        images = np.transpose(images, (0, 3, 1, 2))

    if h_num == -1:
        h_num = images.shape[0]

    line_num = images.shape[0] // h_num
    rest_num = images.shape[0] - line_num * h_num

    image_data = []

    width = images.shape[-1]
    height = images.shape[-2]
    for i in range(line_num):
        image_line = np.zeros((images.shape[1],height,width*h_num))
        for j in range(h_num):
            image_line[:,:,j*width:(j+1)*width] = images[h_num*i + j]
        image_data.append(image_line)

    if rest_num > 0:
        last_line = np.zeros((images.shape[1],height,width*rest_num))

        for j in range(rest_num):
            last_line[:,:,j*width:(j+1)*width] = images[h_num*line_num + j]

        if line_num > 0:
            white = np.zeros((images.shape[1],height,width*(h_num-rest_num)))
            last_line = np.concatenate([last_line,white],axis=-1)

        image_data.append(last_line)
    n_image = np.concatenate(image_data,axis=1)
    n_image = n_image.transpose((1,2,0))
    # cv2.imwrite('test1.png', n_image[:,:,0])

    isize = (int(n_image.shape[1]*factor), int(n_image.shape[0]*factor))
    n_image = cv2.resize(n_image,isize)
    if len(n_image.shape) == 2:
        n_image = np.expand_dims(n_image,2)
    # cv2.imwrite('test2.png',n_image)
    if out_type == 'CHW':
        n_image = n_image.transpose((2,0,1))
    n_image = n_image.astype('uint8')
    return n_image

def crop_image_black(img, move_step = 2):
    tmp_img = img
    if len(img.shape) == 3:
        tmp_img = img[0]

    img_size = (tmp_img.shape[1], tmp_img.shape[0])
    center_point = (tmp_img.shape[1] // 2, tmp_img.shape[0] // 2)

    width_range = []
    for turn in [-1,1]:
        cur_point = center_point[0]
        while cur_point > 0 and cur_point < img_size[0]:
            tmp_list = tmp_img[center_point[1] - 20: center_point[1] + 20, cur_point]
            if (tmp_list < 0.5).astype('uint8') / tmp_list.shape[0] > 0.5:
                width_range.append(cur_point)
                break
            cur_point += turn * move_step

        if cur_point < 0:
            cur_point = 0
            width_range.append(cur_point)

        if cur_point > img_size[0]:
            cur_point = img_size[0]
            width_range.append(cur_point)

    height_range = []
    for turn in [-1,1]:
        cur_point = center_point[1]
        while cur_point > 0 and cur_point < img_size[1]:
            tmp_list = tmp_img[cur_point, center_point[0] - 20: center_point[0] + 20]
            if (tmp_list < 0.5).astype('uint8') / tmp_list.shape[0] > 0.5:
                height_range.append(cur_point)
                break
            cur_point += turn * move_step

        if cur_point < 0:
            cur_point = 0
            height_range.append(cur_point)

        if cur_point > img_size[1]:
            cur_point = img_size[1]
            height_range.append(cur_point)

    return img[height_range[0]:height_range[1], width_range[0]:width_range[1]]

def darw_txt_on_image(img, txt, text_size = 50, loc = 'top'):
    text_image = np.ones((int(text_size * 1.2), img.shape[1], 3)) * 255
    text_image = text_image.astype('uint8')
    text_image = Image.fromarray(text_image)
    draw = ImageDraw.Draw(text_image)
    # fontStyle = ImageFont.truetype("util/NotoSansCJK-Bold.ttc", 80, encoding="utf-8")
    fontStyle = ImageFont.truetype("util/NotoSansCJK-Bold-6.otf", text_size, encoding="utf-8")

    draw.text((10, 10), txt, (0,0,0), font=fontStyle)

    text_image = np.array(text_image)
    if loc == 'top':
        tmp_image = np.concatenate([text_image, img], axis=0)
    else:
        tmp_image = np.concatenate([img, text_image], axis=0)
    return tmp_image


def save_debug_image(img, img_dir, norm2_255 = False, ap = ''):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    mkdir(img_dir)
    fs = [f for f in os.listdir(img_dir) if f.startswith(ap)]
    img_num = len(fs)
    if norm2_255:
        img = img - np.min(img).item()
        img *= (255 / np.max(img).item())
    p = os.path.join(img_dir, ap + '_' + str(img_num) + '.jpg')
    print('save',p)
    cv2.imwrite(p, img.astype('uint8'))
