import cv2
import copy
from .imgaug import augmenters as iaa
import torch
import numpy as np 
import random
from skimage import transform as sktsf

def wc_transform(img, wc=None, ww=None):
    if wc != None:
        wmin = (wc*2 - ww) // 2
        wmax = (wc*2 + ww) // 2
    else:
        wmin = img.min()
        wmax = img.max()
    dfactor = 255.0 / (wmax - wmin)
    img = np.where(img < wmin, wmin, img)
    img = np.where(img > wmax, wmax, img)
    img = (img - wmin) * dfactor
    img = img.astype(np.float32)
    d, _, _ = img.shape
    img = sktsf.resize(img, (d, 256, 256), anti_aliasing=True)

    return img

def gen_ind(fnum, ex_fnum=300, is_valid = True):
    if ex_fnum > fnum:
        cp_time = ex_fnum // fnum

        rest_num = ex_fnum - fnum * cp_time
        if rest_num > 0:
            cp_inter = fnum // rest_num

        f_inds = []

        for i in range(fnum):
            for _ in range(cp_time):
                f_inds.append(i)
                if rest_num > 0 and i % cp_inter == 0:
                    f_inds.append(i)
                    rest_num -= 1

    elif ex_fnum < fnum:
        f_inds = []
        rm_times = fnum // ex_fnum

        if is_valid:
            rand_int = 0
        else:
            rand_int = random.randint(0, rm_times - 1)

        for i in range(fnum):
            if i % rm_times == rand_int:
                f_inds.append(i)
            if (i + 1) % rm_times == 0 and not is_valid:
                rand_int = random.randint(0, rm_times - 1)

        n_fnum = len(f_inds)
        rest_num = n_fnum - ex_fnum

        if rest_num > 0:
            if is_valid:
                rm_inter = n_fnum // rest_num
                i = n_fnum - 1
                c = 0
                while i >= 0:
                    del f_inds[i]
                    # print(i)
                    i -= rm_inter
                    c += 1
                    if i < 0 or c == rest_num:
                        break
            else:
                de_inds = random.sample(list(range(n_fnum)), rest_num)
                de_inds = sorted(de_inds, reverse=True)
                for ind in de_inds:
                    del f_inds[ind]
    else:
        f_inds = list(range(fnum))

    return f_inds

def resize_frames(frames, ex_fnum = 30, is_valid = True):
    fnum = len(frames)
    a_inds = gen_ind(fnum=fnum,ex_fnum=ex_fnum, is_valid = is_valid)
    new_frames = [frames[ind] for ind in a_inds]
    if isinstance(frames, np.ndarray):
        new_frames = np.stack(new_frames, axis=0)
    return new_frames

def bbb_index(imgs):
    index_list = []
    for i in range(imgs.shape[0]):
        # if np.sum((imgs[i] > 0).astype('int')) > 10:
        if np.any(imgs[i]):
            index_list.append(i)
    return index_list

def bbb(imgs):
    '''
    bbb is short of body bound box
    :param data_dict:
    :return:
    '''

    for i in range(imgs.shape[0]):
        assert np.any(imgs[i])

    def enlarge_box(box1, box2):
        bbox = [0] * 4
        bbox[0] = min(box1[0], box2[0])
        bbox[1] = min(box1[1], box2[1])
        bbox[2] = max(box1[2], box2[2])
        bbox[3] = max(box1[3], box2[3])
        return bbox

    def tmp_crop(im, d):
        #get bounding boxPoints
        cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_list = sorted(cnts, key=cv2.contourArea, reverse=True)

        res = []
        for c in c_list:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            res.append([x, y, x + w, y + h])

        bbox = res[0]
        for tmp_bbox in res[1:]:
            bbox = enlarge_box(bbox, tmp_bbox)
            # bbox[0] = min(bbox[0], tmp_bbox[0])
            # bbox[1] = min(bbox[1], tmp_bbox[1])
            # bbox[2] = max(bbox[2], tmp_bbox[2])
            # bbox[3] = max(bbox[3], tmp_bbox[3])

        return bbox

    boxes_list = []
    boxes_data = []
    for i in range(imgs.shape[0]):
        res = tmp_crop(imgs[i], i)
        if len(res) > 4:
            print('something wrong')
        boxes_list.append(res)
        boxes_data.append([copy.copy(res)])

    cur_box = boxes_list[0]
    box_info = []
    sd = 0
    ed = 0
    for box in boxes_list[1:]:
        ed += 1
        if compute_IOU(cur_box, box) > 0.1:
            cur_box = enlarge_box(cur_box, box)
        else:
            cur_box += [sd, ed]
            box_info.append(cur_box)
            cur_box = box
            sd = ed

    # if len(box_info) == 0 or sd != ed:
    cur_box += [sd, ed]
    box_info.append(cur_box)

    return box_info, boxes_data

def expand_bbox(bbox, img_shape, ex_ratio, certain_pixel = -1):
    '''
    bbox: x1, y1, x2, y2
    '''

    if ex_ratio <= 0 and certain_pixel <= 0:
        x1 = 0
        x2 = img_shape[0] - 1
        y1 = 0
        y2 = img_shape[1] - 1
        return [int(x1), int(y1), int(x2), int(y2)]

    p1 = np.array([bbox[0], bbox[1]]).astype('float32')
    p2 = np.array([bbox[2], bbox[3]]).astype('float32')
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

    c_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])


    if certain_pixel <= 0:
        p1 -= c_point
        p2 -= c_point

        p1 *= ex_ratio
        p2 *= ex_ratio

        p1 += c_point
        p2 += c_point

    else:
        nx1 = c_point[0] - certain_pixel // 2
        ny1 = c_point[1] - certain_pixel // 2
        nx2 = c_point[0] + certain_pixel // 2
        ny2 = c_point[1] + certain_pixel // 2

        x1 = min(nx1, x1)
        y1 = min(ny1, y1)
        x2 = max(nx2, x2)
        y2 = max(ny2, y2)

        p1 = [x1,y1]
        p2 = [x2,y2]

    x1 = max(p1[0], 0)
    x2 = min(p2[0], img_shape[0])
    y1 = max(p1[1], 0)
    y2 = min(p2[1], img_shape[1])


    return [int(x1), int(y1), int(x2), int(y2)]

def generate_mask(img_shape, bbox_data):
    '''
    bbox: B * N * [x1, y1, x2, y2]
    '''

    mask = np.zeros((img_shape)).astype('uint8')
    for i, bbox_list in enumerate(bbox_data):
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox
            mask[i, y1: y2 + 1, x1: x2 + 1] = 255
    return mask

def compute_IOU(rec1,rec2):
    left_column_max = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max = max(rec1[1],rec2[1])
    down_row_min = min(rec1[3],rec2[3])
    
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/ min(S1, S2)

def box_in_box(bbox1, bbox2):
    if bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]:
        return True
    else:
        return False

def norm_data(x, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), ctype = 'CHW', norm_to_one = True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    x = x.float()

    if ctype == 'HWC':
        if len(x.shape) == 3:
            x = x.permute(2,0,1)
        elif len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)

    if norm_to_one: 
        x = x / 255.
    # if x.shape[0] != 3 and x.shape[0] != 4:
    #     assert False, 'The dim is not right'

    x = normalize(x, mean, std)
    return x

def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    # if not _is_tensor_image(tensor):
    #     raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if len(tensor.shape) == 3:
        tmp_mean = mean[:, None, None]
        tmp_std = std[:, None, None]
    elif len(tensor.shape) == 4:
        tmp_mean = mean[None, :, None, None]
        tmp_std = std[None, :, None, None]
    else:
        raise ValueError()

    tensor.sub_(tmp_mean).div_(tmp_std)
    return tensor

def trans_bboxes(old_bboxes_data):
    bboxes_data = copy.deepcopy(old_bboxes_data)
    max_len = 1
    new_bboxes = []

    for seq_bbox_data in bboxes_data:
        new_seq_bbox_data = []
        for tmp_bboxes in seq_bbox_data:
            if len(tmp_bboxes) > max_len:
                tmp_bboxes = tmp_bboxes[:max_len]
            if len(tmp_bboxes) < max_len:
                while len(tmp_bboxes) < max_len:
                    tmp_bboxes.append([-1,-1,-1,-1])
            new_seq_bbox_data.append(tmp_bboxes)
        new_bboxes.append(new_seq_bbox_data)
    new_bboxes = torch.Tensor(new_bboxes)
    return new_bboxes
