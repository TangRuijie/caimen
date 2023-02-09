# federate
from __future__ import  absolute_import
from __future__ import  division
import numpy as np
import os
import copy
import re
import cv2
from .base_dataset import BaseDataset, default_collate
from skimage import transform as sktsf
import random
import json
from util.basic import *
from util.video import resize_frames
from util.image import save_debug_image

class ZGFullTumorDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--with_full_image', type=int, default=0)
        parser.add_argument('--load_bbox', type=int, default=0)
        parser.add_argument('--resize_dcm_len', type=int, default=20)
        parser.add_argument('--window_list', type=str, default='[[0,400],[0,400],[0,400]]')
        parser.add_argument('--certain_crop_size', type=int, default=128)
        parser.add_argument('--bbox_ex_ratio1', type=float, default=1.5)
        parser.add_argument('--bbox_ex_ratio2', type=float, default=2)
        parser.add_argument('--zengqiang', type=int, default=0)
        parser.add_argument('--target_focus', type=int, default=0)
        parser.add_argument('--multi_task', type=int, default=0)
        parser.add_argument('--multi_task_ratio', type=float, default=1)
        return parser

    def get_location_label(self, location):
        if location == "front":
            return 0
        elif location == "intermediate":
            return 1
        elif location == "behind":
            return 2
        else:
            print(f"wrong, the label is {location}")
            return 3

    def __init__(self, opt, data_type):
        super().__init__(opt, data_type)
        self.img_aug = BaseDataset.get_aug_transform(opt, in_type='NHW', out_type='NHW')
        self.img_resize = BaseDataset.get_aug_transform(opt, preprocess='resize', in_type='NHW', out_type='NHW')
        self.data_type = data_type
        if isinstance(self.opt.window_list, str):
            self.opt.window_list = json.loads(opt.window_list)

        self.window_list = self.opt.window_list
        self.target_indexs = []
        print('data_type ', data_type)

        self.path_list = []
        self.label_list = []

        # multi-classification
        hospital_list = ['sx', 'cc', 'sz', 'zs', 'nm', 'sc', 'gy', 'gzzl', 'zd', 's3', 'fj', 'xa', 'ln', 'cq', 'qd']  

        tree_construct_dict = {
            '0': ['thymoma', 'benign cyst', 'thymic carcinoma', 'germ cell tumor', 'other soft tissue tumor', 'neuroendocrine tumor', 'thymic hyperplasia', 'lymphoma', 'lymphadenosis', 'ectopicthyroidgland', 'granulomatous inflammation', 'neurogenic tumor'],
            '00': ['thymoma', 'benign cyst', 'thymic carcinoma', 'germ cell tumor', 'neuroendocrine tumor', 'thymic hyperplasia', 'lymphoma', 'ectopicthyroidgland'],
            '01': ['other soft tissue tumor', 'lymphadenosis', 'granulomatous inflammation', 'neurogenic tumor'],
            '000': ['thymoma', 'thymic carcinoma', 'neuroendocrine tumor', 'lymphoma'],
            '001': ['benign cyst', 'germ cell tumor', 'thymic hyperplasia', 'ectopicthyroidgland'],
            '010': ['other soft tissue tumor', 'neurogenic tumor'],
            '011': ['lymphadenosis', 'granulomatous inflammation'],
            '0000': ['thymoma', 'thymic carcinoma', 'neuroendocrine tumor'],
            '0010': [],
            '0011': [],
            '00000': []
        }
        tree_training_dict = {
            '0': [['thymoma', 'benign cyst', 'thymic carcinoma', 'germ cell tumor', 'neuroendocrine tumor', 'thymic hyperplasia', 'lymphoma', 'ectopicthyroidgland'], ['other soft tissue tumor', 'lymphadenosis', 'granulomatous inflammation', 'neurogenic tumor']],
            '00': [['thymoma', 'thymic carcinoma', 'neuroendocrine tumor', 'lymphoma'], ['benign cyst', 'germ cell tumor', 'thymic hyperplasia', 'ectopicthyroidgland']],
            '01': [['other soft tissue tumor', 'neurogenic tumor'], ['lymphadenosis', 'granulomatous inflammation']],
            '000': [['thymoma', 'thymic carcinoma', 'neuroendocrine tumor'], ['lymphoma']],
            '001': [['benign cyst', 'thymic hyperplasia'], ['germ cell tumor', 'ectopicthyroidgland']],
            '010': [['other soft tissue tumor'], ['neurogenic tumor']], #
            '011': [['lymphadenosis'], ['granulomatous inflammation']], #
            '0000': [['thymoma', 'thymic carcinoma'], ['neuroendocrine tumor']],
            '0010': [['benign cyst'], ['thymic hyperplasia']],
            '0011': [['germ cell tumor'], ['ectopicthyroidgland']],
            '00000': [['thymoma'], ['thymic carcinoma']] #
        }
        
        self.disease_list = tree_construct_dict[opt.path]
        self.subgroup_list = tree_training_dict[opt.path]
        data_dict = json.load(open(f'buffer/federate/{opt.path}/{hospital_list[opt.index]}_{data_type}.json'))

        with open("class_label.json", "r") as f:
            self.multi_task_info = json.load(f)
        
        for label, tmp_list in data_dict.items():
            self.path_list.extend(tmp_list)
            if self.opt.target_focus < 0:
                self.label_list.extend([int(label)] * len(tmp_list))
            else:
                if int(label) == self.opt.target_focus:
                    self.label_list.extend([1] * len(tmp_list))
                else:
                    self.label_list.extend([0] * len(tmp_list))

        assert len(self.label_list) == len(self.path_list)

        if self.data_type == 'train' or self.opt.l_state != 'train':
            if self.opt.target_focus < 0:
                self.opt.class_num = len(data_dict.keys())
            else:
                self.opt.class_num = 2

            tmp_sample_list = [0] * self.opt.class_num

            for label in self.label_list:
                tmp_sample_list[label] += 1

            self.opt.sample_num_list = tmp_sample_list

        assert len(self.path_list) == len(self.label_list)

    def load_item(self, index):
        path = self.path_list[index]
        id = path.split('/')[-1].replace('.npz', '')
        item_info = self.multi_task_info[id]
        hospital, disease_type, location, age, gender = item_info.split()
        
        # tree training
        stored = False
        for idx, disease_list in enumerate(self.subgroup_list):
            if disease_type in disease_list:
                label = idx
                stored = True
                break
        if not stored:
            label = 0

        # tree construct
        #try:
        #    label = self.disease_list.index(disease_type)
        #except:
        #    label = 0

        label2 = 0
        if gender == 'M':
            gender = 0
        else:
            gender = 1

        data = np.load(path)
        try:
            CT = data['data'] # h, w, d
            mask = data['labels'] # d, h, w
        except:
            CT = data['CT']
            mask = data['mask']
        CT = CT.transpose((2,0,1))
        
        #loc = self.get_location_label(location) % 4
        #age = eval(age.strip('0').replace('Y', ''))
        loc = 0
        age = 0
        
        mask = (mask / np.max(mask)) * 255
        mask = mask.astype('uint8')

        wl, ww = self.window_list[0]
        if ww is None or ww < 0:
            CT = self.wc_transform(CT, None, None)
        else:
            if CT.min() >= 0 and CT.max() <= 255:
                CT = CT.astype(np.float32)
            else:
                CT = self.wc_transform(CT, wl, ww)

        bbox_idxes = bbb_index(mask)
        if CT.shape != mask.shape:
            return None

        if len(bbox_idxes):
            CT = CT[bbox_idxes]
            mask = mask[bbox_idxes]
        else:
            print('empty ')
            return None

        img_shape = [CT.shape[-1], CT.shape[-2]]

        max_bboxes, or_bboxes = bbb(mask)

        bbox_data = []
        CT = self.img_resize(CT)
        mask = self.img_resize(mask)

        if len(max_bboxes) >= 2:
            max_bbox = max_bboxes[0]
            d1 = max_bbox[-2]
            d2 = max_bbox[-1]
            max_len = d2 - d1
            max_ind = 0
            for i in range(1, len(max_bboxes)):
                max_bbox = max_bboxes[i]
                d1 = max_bbox[-2]
                d2 = max_bbox[-1]
                tmp_len = d2 - d1
                if tmp_len > max_len:
                    max_ind = i
                    max_len = tmp_len
            max_bboxes = max_bboxes[max_ind: max_ind + 1]


        if len(max_bboxes) == 1:
            # max_bboxes = sorted(max_bboxes, key=lambda x:x[-1] - x[-2])
            x1, y1, x2, y2, d1, d2 = max_bboxes[0]
            crop_size = expand_bbox([x1, y1, x2, y2], img_shape, self.opt.bbox_ex_ratio2, certain_pixel=self.opt.certain_crop_size)
            c_x1, c_y1, c_x2, c_y2 = crop_size
            CT = CT[d1: d2 + 1, c_y1:c_y2+1, c_x1:c_x2+1]
            mask = mask[d1: d2 + 1, c_y1:c_y2+1, c_x1:c_x2+1]
            or_bboxes = or_bboxes[d1:d2 + 1]

            tmp_inds = []
            for i, bboxes in enumerate(or_bboxes):
                tmp_bboxes = []
                for bbox in bboxes:
                    ex_bbox = expand_bbox(bbox, img_shape, self.opt.bbox_ex_ratio1)
                    if box_in_box(ex_bbox, [c_x1, c_y1, c_x2, c_y2]):
                        ex_bbox[0] -= c_x1
                        ex_bbox[2] -= c_x1
                        ex_bbox[1] -= c_y1
                        ex_bbox[3] -= c_y1
                        tmp_bboxes.append(ex_bbox)

                if len(tmp_bboxes):
                    bbox_data.append(tmp_bboxes)
                    tmp_inds.append(i)

            if len(tmp_inds) == 0:
                return None

            CT = CT[tmp_inds]
            mask = mask[tmp_inds]

        else:
            # print('max bboxes len', len(max_bboxes))
            # print(index)

            new_img_list = []
            img_path = 'vis/tumor/' + str(index) + '_cat_' + str(label) + '.jpg'
            # print(img_path)
            for max_ind, max_bbox in enumerate(max_bboxes):
                x1, y1, x2, y2, d1, d2 = max_bboxes[max_ind]
                max_bbox = [x1, y1, x2, y2]
                color = [0] * 3
                color[max_ind % 3] = 255
                print('max ind',max_ind, 'd1, d2',d1,d2, max_bbox)

                CT = self.img_resize(CT)
                for i in range(d1, d2 + 1):
                    tmp_img = CT[i,...]
                    or_box_list = or_bboxes[i]
                    tmp_img = np.repeat(tmp_img[..., np.newaxis], axis = -1, repeats=3)
                    tmp_img = tmp_img.astype('uint8')
                    # for bbox in bbox_list:
                    tmp_img = cv2.rectangle(tmp_img, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), color=color, thickness=1)
                    for bbox in or_box_list:
                        tmp_img = cv2.rectangle(tmp_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,0,255), thickness=1)

                    new_img_list.append(tmp_img)
            new_img = np.concatenate(new_img_list, axis=1)
            cv2.imwrite(img_path, new_img)
            return None
            # print('bbox ', bbox_data)

        #mask = generate_mask(CT.shape, bbox_data)

        target_inds = resize_frames(list(range(CT.shape[0])), self.opt.resize_dcm_len, is_valid=(self.data_type != 'train'))
        CT = CT[target_inds]
        mask = mask[target_inds]

        CT = np.concatenate([mask, CT], axis=0)
        try:
            CT = self.img_aug(CT)
        except:
            print('max bbox',max_bboxes[0])
            print('CT shape', CT.shape)
            print('crop size',crop_size)
            print('id', path)
            return None

        mask = CT[:self.opt.resize_dcm_len].astype('uint8')
        CT = CT[self.opt.resize_dcm_len:]
        try:
            _, bbox_data = bbb(mask)
        except:
            return None

        CT = np.stack([CT]*3, axis=1)
        heatmap = -1

        CT = BaseDataset.norm_data(CT, norm_to_one=True)

        mask = torch.Tensor(mask)
        mask = mask / torch.max(mask)
        mask = mask.unsqueeze(1)

        loc_mask = loc * torch.ones_like(mask)
        age_mask = age * torch.ones_like(mask)
        if self.opt.in_channel == 6:
            CT = torch.cat([CT, mask, loc_mask/4, age_mask], dim = 1)
        elif self.opt.in_channel == 4:
            CT = torch.cat([CT, mask], dim = 1)
        return CT, label, label2, path, heatmap, loc, age, gender, mask, bbox_data

    def __getitem__(self, index):
        data = None
        while data is None:
            # print('index ', index)
            data = self.load_item(index)
            if data is not None:
                return data
            index = (index + 1) % len(self.path_list)

    def __len__(self):
        return len(self.path_list)

    @staticmethod
    def collect_fn(batch):
        part1 = [bt[:-1] for bt in batch]
        part2 = [bt[-1] for bt in batch]
        part1 = default_collate(part1)
        part1.append(part2)
        return part1

    def wc_transform(self, img, wc=None, ww=None, norm2one = False):
        if wc != None:
            wmin = (wc*2 - ww) // 2
            wmax = (wc*2 + ww) // 2
        else:
            wmin = img.min()
            wmax = img.max()
        if norm2one:
            dfactor = 1.0 / (wmax - wmin)
        else:
            dfactor = 255.0 / (wmax - wmin)
        img = np.where(img < wmin, wmin, img)
        img = np.where(img > wmax, wmax, img)
        img = (img - wmin) * dfactor
        img = img.astype(np.float32)
        d, h, w = img.shape

        return img

def box_in_box(bbox1, bbox2):
    if bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]:
        return True
    else:
        return False

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