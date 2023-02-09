import os
import cv2
import sys
sys.path.insert(0, '.')
import os.path as osp
# from util.basic import *
from util.image import darw_txt_on_image
vis_dir = 'vis/liang'
save_dir = 'vis/liang_text'
legend_list = ['胸腺瘤', '良性囊肿', '神经源性肿瘤', '胸腺癌', '纵隔生殖细胞瘤', '纵隔软组织肿瘤', '淋巴瘤', '神经内分泌肿瘤', '淋巴组织增生', '异位甲状腺肿瘤', '胸腺增生', '肉芽肿性炎']

fs = os.listdir(vis_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for f in fs:
    seq_ind, label, pred = f.split('_')[:3]
    img = cv2.imread(osp.join(vis_dir, f))

    tmp_text = legend_list[label]  + ' 被分为 ' + legend_list[pred]
    darw_txt_on_image(img, tmp_text)
    save_path = os.path.join(save_dir, f)
    cv2.imwrite(save_path, img)
