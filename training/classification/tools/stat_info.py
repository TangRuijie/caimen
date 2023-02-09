import copy
import os
import json
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
from util.basic import *
from util.image import darw_txt_on_image

def wc_transform(img, wc=None, ww=None, norm2one = False):
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

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 占小面积比例
    """
    left_column_max = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max = max(rec1[1],rec2[1])
    down_row_min = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/ min(S1, S2)

def bbb_index(imgs):
    index_list = []
    for i in range(imgs.shape[0]):
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
        boxes_list.append(res)
        boxes_data.append([res])

    cur_box = boxes_list[0]
    box_info = []
    sd = 0
    ed = 0
    for box in boxes_list[1:]:
        if compute_IOU(cur_box, box) > 0.1:
            cur_box = enlarge_box(cur_box, box)
            ed += 1
        else:
            cur_box += [sd, ed]
            box_info.append(cur_box)
            cur_box = box
            sd = ed

    if sd == 0 or sd != ed:
        cur_box += [sd, ed]
        box_info.append(cur_box)

    return box_info, boxes_data

json_path = 'checkpoints/Apr09_08-49stage0,zengqiang=2,loss_type=balanced,gpu_ids=0_52.727/optimal_valid_buffer.json'
legend_list = ['胸腺瘤', '良性囊肿', '神经源性肿瘤', '胸腺癌', '纵隔生殖细胞瘤', '纵隔软组织肿瘤', '淋巴瘤', '神经内分泌肿瘤', '淋巴组织增生', '异位甲状腺肿瘤', '胸腺增生', '肉芽肿性炎']
name_list = []

stat_dict = {}

buf_dict = json.load(open(json_path))

label_list = buf_dict['glabels']
pred_list = buf_dict['gpreds']
path_list = buf_dict['ginput_ids']

window_list = [[0,400],[0,400],[0,400]]

for i, (label, pred) in enumerate(zip(label_list, pred_list)):
    if label not in stat_dict:
        stat_dict[label] = [[], []]

    if label == pred:
        stat_dict[label][0].append((i, label))
    else:
        stat_dict[label][1].append((i, pred))


for label in stat_dict.keys():
    for tp in range(2):
        tmp_list = stat_dict[label][tp][:5]



        for seq_ind, item in enumerate(tmp_list):
            ind, pred = item
            path = path_list[item[0]]
            data = np.load(path)
            CT = data['data'].transpose((2,0,1))
            mask = data['labels']
            mask = (mask / np.max(mask)) * 255
            bbox_idxes = bbb_index(mask)
            if len(bbox_idxes):
                CT = CT[bbox_idxes]
                mask = mask[bbox_idxes]
            else:
                print('empty ')
                exit()

            img_shape = [CT.shape[-1], CT.shape[-2]]
            wl, ww = window_list[0]
            if ww is None or ww < 0:
                CT = wc_transform(CT, None, None)
            else:
                CT = wc_transform(CT, wl, ww)

            mask = mask.astype('uint8')
            max_bboxes, or_bboxes = bbb(mask)

            x1, y1, x2, y2, d1, d2 = max_bboxes[0]
            CT = CT[d1: d2 + 1]
            or_bboxes = or_bboxes[d1:d2 + 1]
            CT = np.stack([CT] * 3, axis=-1).astype('uint8')

            ct_list = []
            for i in range(CT.shape[0]):
                bboxes = or_bboxes[i]
                ct = CT[i]
                for bbox in bboxes:
                    ct = cv2.rectangle(ct, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0,255,0), thickness=1)
                ct_list.append(ct)
                if len(ct_list) >= 5:
                    break

            CT = np.concatenate(ct_list, axis=1)
            if tp == 0:
                r = 'right'
            else:
                r = 'wrong'

            vis_dir = 'vis/liang'
            mkdir(vis_dir)
            tmp_name = str(label) + '_' + r + '_' + str(seq_ind) + '_' + str(pred)  + '.jpg'
            save_path = os.path.join(vis_dir, tmp_name)
            tmp_text = legend_list[label]  + ' 被分为 ' + legend_list[pred]
            CT = darw_txt_on_image(CT, tmp_text)
            cv2.imwrite(save_path, CT)