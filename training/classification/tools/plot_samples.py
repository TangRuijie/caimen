import cv2
import sys
sys.path.insert(0,'.')
from util.basic import *

def plot_sample(load_dir = '',  bbox_ex_ratio2 = 5):
    fs = os.listdir(load_dir)
    if 'optimal_valid_buffer.json' in fs:
        buf_dict = json.load(open(osp.join(load_dir, 'optimal_valid_buffer.json')))
    else:
        buf_dict = json.load(open(osp.join(load_dir, 'valid_buffer.json')))

    # self.buffer_ginput_ids = []
    # self.buffer_gscores = []
    # self.buffer_glabels = []
    # self.buffer_gpreds = []

    path_list = buf_dict['ginput_ids']
    score_list = buf_dict['gscores']
    label_list = buf_dict['glabels']
    pred_list = buf_dict['gpreds']

    wrong_dict = {}

    for i, (label, pred, score) in enumerate(zip(label_list, pred_list, score_list)):
        if label != pred:
            if label not in wrong_dict.keys():
                wrong_dict[label] = []
            wrong_dict[label].append((path_list[i], pred, score))

    save_dir = osp.join('vis', 'wrong')
    mkdir(save_dir)
    window_list = [[0,400],[0,400],[0,400]]

    # import ipdb
    # ipdb.set_trace()

    for k, v_data in wrong_dict.items():
        cur_dir = osp.join(save_dir, str(k))
        mkdir(cur_dir)

        for p_idx, (p, pred, score) in enumerate(v_data):
            data = np.load(p)
            CT = data['data'].transpose((2,0,1))
            mask = data['labels']
            mask = (mask / np.max(mask)) * 255
            mask = mask.astype('uint8')

            bbox_idxes = bbb_index(mask)
            if len(bbox_idxes):
                CT = CT[bbox_idxes]
                mask = mask[bbox_idxes]
            else:
                print('empty ')
                return None


            wl, ww = window_list[0]
            CT = wc_transform(CT, wl, ww)
            max_bboxes, or_bboxes = bbb(mask)

            # x1, y1, x2, y2, d1, d2 = max_bboxes[max_ind]
            # bbox = [x1, y1, x2, y2]

            x1, y1, x2, y2, d1, d2 = max_bboxes[0]
            img_shape = [CT.shape[-2], CT.shape[-1]]
            crop_size = expand_bbox([x1, y1, x2, y2], img_shape, bbox_ex_ratio2)

            color = (0,255,0)
            enlarge_color = (0,0,255)
            new_img_list = []
            for i in range(len(or_bboxes)):
                bbox = or_bboxes[i][0]
                tmp_img = CT[i,...]
                tmp_img = np.repeat(tmp_img[..., np.newaxis], axis = -1, repeats=3)
                tmp_img = tmp_img.astype('uint8')
                # for bbox in bbox_list:
                bbox = np.array(bbox).astype('uint16')
                cv2.rectangle(tmp_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=1)
                cv2.rectangle(tmp_img, (crop_size[0], crop_size[1]), (crop_size[2], crop_size[3]), color=enlarge_color, thickness=1)

                tmp_img = cv2.resize(tmp_img, (256,256))
                new_img_list.append(tmp_img)
            new_img = np.concatenate(new_img_list, axis=1)
            save_path = cur_dir + '/' + str(p_idx) + '_cat_' + str(pred) + '.jpg'
            print(save_path)
            cv2.imwrite(save_path, new_img)


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
    # return boxes_data



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


def expand_bbox(bbox, img_shape, ex_ratio):
    '''
    bbox: x1, y1, x2, y2
    '''
    p1 = np.array([bbox[0], bbox[1]]).astype('float32')
    p2 = np.array([bbox[2], bbox[3]]).astype('float32')
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

    c_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    p1 -= c_point
    p2 -= c_point

    p1 *= ex_ratio
    p2 *= ex_ratio

    p1 += c_point
    p2 += c_point

    x1 = max(p1[0], 0)
    x2 = min(p2[0], img_shape[0])
    y1 = max(p1[1], 0)
    y2 = min(p2[1], img_shape[1])

    return [int(x1), int(y1), int(x2), int(y2)]

if __name__ == '__main__':
    load_dir = 'checkpoints/Apr06_11-18TEST_stage0,model=transformer,dataset_mode=zgheattumorV3,v_dataset_mode=zgheattumorV3,gpu_ids=0_36.364'

    plot_sample(load_dir)
