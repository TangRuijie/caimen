import torch.nn.functional as F
import torch
import numpy as np
from .networks.fpn_transformer_net import ResNet as FPNNet
from .utils import *

def generate_classification_input(CT, mask):
    mask = (mask / np.max(mask)) * 255
    mask = mask.astype('uint8')
    CT = wc_transform(CT, 0, 400)

    # take the ct slice with lesion segmentation
    bbox_idxes = bbb_index(mask)

    CT = CT[bbox_idxes]
    mask = mask[bbox_idxes]

    img_shape = [CT.shape[-1], CT.shape[-2]]
    max_bboxes, or_bboxes = bbb(mask)

    bbox_data = []

    # select the largest bounding box
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

    x1, y1, x2, y2, d1, d2 = max_bboxes[0]
    crop_size = expand_bbox([x1, y1, x2, y2], img_shape, 2, certain_pixel=192)
    c_x1, c_y1, c_x2, c_y2 = crop_size
    CT = CT[d1: d2 + 1, c_y1:c_y2+1, c_x1:c_x2+1]
    mask = mask[d1: d2 + 1, c_y1:c_y2+1, c_x1:c_x2+1]
    or_bboxes = or_bboxes[d1:d2 + 1]

    tmp_inds = []
    for i, bboxes in enumerate(or_bboxes):
        tmp_bboxes = []
        for bbox in bboxes:
            ex_bbox = expand_bbox(bbox, img_shape, 1.2)
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
        print('wrong')

    CT = CT[tmp_inds]
    mask = mask[tmp_inds]

    target_inds = resize_frames(list(range(CT.shape[0])), 20, is_valid=True)
    CT = CT[target_inds]
    mask = mask[target_inds]

    CT = np.concatenate([mask, CT], axis=0)

    mask = CT[:20].astype('uint8')
    CT = CT[20:]
    _, bbox_data = bbb(mask)

    CT = np.stack([CT]*3, axis=1)

    CT = norm_data(CT, norm_to_one=True)

    mask = torch.Tensor(mask)
    mask = mask / torch.max(mask)
    mask = mask.unsqueeze(1)

    output = torch.cat([CT, mask], dim = 1)
    output = output.cuda().float()
    bbox_data = trans_bboxes(bbox_data)
    mask = mask.cuda().float()
    output = torch.unsqueeze(output, 0)
    bbox_data = torch.unsqueeze(bbox_data, 0)
    mask = torch.unsqueeze(mask, 0)
    
    return output, bbox_data, mask

def load_classification_model():
    params_dict = {
        'fpn': {
            'in_channel': 4,
            'batch_size': 1,
            'load_size': 256,
            'seq_len': 20,
            'embd_pos': 0,
            'heat_layer': 4,
            'up_layer_size': 8,
            'crop_ratio': 0.3,
            'context_frames': 3,
            'roi_op': 'directly',
            'add_info': '000',
            'fpn_channels': 256,
            'head_block_list': [0,1,1,1],
            'frame_comb_mode': 'mean',
            'channel_comb_type': 'add',
            'block_comb_mode': 'cat',
            'head_comb_mode': 'cat',
            'sub_d_model': 128,
            'block_layer': 2,
            'multi_task': False,
            'head_number': 2
        }
    }
    net_res = FPNNet(net_name = "ResNet50", bottleneck_dim = 256,
                    class_num = 12, **params_dict['fpn'])
    model_dict = torch.load('cls_module/model_weights/cls_model.pth')
    net_res.load_state_dict(model_dict)
    net_res.cuda()
    net_res.eval()
    
    return net_res

def ai_classification(ct, mask):
    cls_model = load_classification_model()
    ct_with_mask, bbox, mask = generate_classification_input(ct, mask)
    tmp_y, _ = cls_model(ct_with_mask, bbox, mask)
    pred = F.softmax(tmp_y, dim = 1)

    return pred.cpu().detach().numpy()[0]