# coding=UTF-8
import json
import os
import numpy as np
import cv2

data_root_path = '/GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA'
split_path = '/GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA/Classification/split'
# /GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA/noCrop/data

def make_label_dict():
    label_dict = {}
    for r, ds, fs in os.walk(split_path):
        fs = [f for f in fs if f.lower().endswith('train.txt') or f.lower().endswith('val.txt')]
        for fname in fs:
            fpath = os.path.join(r, fname)
            with open(fpath) as f:
                lines = f.readlines()
                for line in lines:
                    name, label = line.lstrip().rstrip().split()
                    name = name.split('_')[0]
                    label_dict[name] = label
    return label_dict

train_list = []
val_list = []
for r, ds, fs in os.walk(os.path.join(data_root_path, "noCrop/data")):
    if len(fs):
        tmp_dir = r
        ids = [tmp_dir + '/' + f for f in fs]
        if r.endswith('_train'):
            train_list.extend(ids)
        elif r.endswith('_val'):
            val_list.extend(ids)





# train_list = []
# for _, _, files in os.walk(os.path.join(data_root_path, "noCrop/data", "SX_train")):
#     for f in files:
#         id, _ = f.split(".")
#         train_list.append(id.strip())
# val_list = []
# for _, _, files in os.walk(os.path.join(data_root_path, "noCrop/data", "SX_val")):
#     for f in files:
#         id, _ = f.split(".")
#         val_list.append(id.strip())
# train_list.sort()
# with open(os.path.join(data_root_path, 'Classification', "class_label.json"), "r") as f:
#     data = json.load(f)

label_list = ['异位甲状腺肿瘤', '胸腺癌', '纵隔软组织肿瘤', '胸腺瘤', '胸腺增生', '良性囊肿', '神经源性肿瘤', '淋巴瘤', '淋巴组织增生', '神经内分泌肿瘤', '纵隔生殖细胞瘤']
train_file = open("data/full_trainV3.txt", "w")
val_file = open("data/full_valV3.txt", "w")

label_dict = make_label_dict()

miss_count = 0
for id in train_list:
    name = os.path.basename(id).replace('.npy', '')
    try:
        class_label = label_dict[name]
    except:
        miss_count += 1
        print('miss count', miss_count)
        continue
    p = id
    # class_label = label_list.index(label)
    train_file.write(p + ' ' + str(class_label) + '\n')


for id in val_list:

    name = os.path.basename(id).replace('.npy', '')
    try:
        class_label = label_dict[name]
    except:
        miss_count += 1
        print('miss count', miss_count)
        continue

    p = id
    # class_label = label_list.index(label)
    val_file.write(p + ' ' + str(class_label) + '\n')

