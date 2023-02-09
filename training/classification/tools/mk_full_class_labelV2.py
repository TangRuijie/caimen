# coding=UTF-8
import json
import os
import numpy as np
import cv2

data_root_path = '/GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA'
# /GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA/noCrop/data

train_list = []
for _, _, files in os.walk(os.path.join(data_root_path, "noCrop/data", "SX_train")):
    for f in files:
        id, _ = f.split(".")
        train_list.append(id.strip())
val_list = []
for _, _, files in os.walk(os.path.join(data_root_path, "noCrop/data", "SX_val")):
    for f in files:
        id, _ = f.split(".")
        val_list.append(id.strip())
train_list.sort()
with open(os.path.join(data_root_path, 'Classification', "class_label.json"), "r") as f:
    data = json.load(f)

label_list = ['异位甲状腺肿瘤', '胸腺癌', '纵隔软组织肿瘤', '胸腺瘤', '胸腺增生', '良性囊肿', '神经源性肿瘤', '淋巴瘤', '淋巴组织增生', '神经内分泌肿瘤', '纵隔生殖细胞瘤']
train_file = open("data/full_trainV2.txt", "w")
val_file = open("data/full_valV2.txt", "w")

for id in train_list:
    idx = str(id).lstrip("0")
    try:
        label = data[idx]
    except:
        continue
    # p = os.path.join(data_root_path, "data_crop", "train", f"{id}.npy")
    p = os.path.join(data_root_path, "noCrop/data", "SX_train", f"{id}.npy")
    # CT = np.load(os.path.join(data_root_path, "data_crop", "train", f"{id}.npy"))
    class_label = label_list.index(label)
    train_file.write(p + ' ' + str(class_label) + '\n')

for id in val_list:
    idx = str(id).lstrip("0")
    try:
        label = data[idx]
    except:
        continue
    # p = os.path.join(data_root_path, "data_crop", "val", f"{id}.npy")
    p = os.path.join(data_root_path, "noCrop/data", "SX_val", f"{id}.npy")
    # CT = np.load(os.path.join(data_root_path, "data_crop", "val", f"{id}.npy"))
    class_label = label_list.index(label)
    val_file.write(p + ' ' + str(class_label) + '\n')
