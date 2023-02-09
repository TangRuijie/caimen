import json
import os
import numpy as np
import cv2

def crop(im, d):
    #get bounding boxPoints
    cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_list = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    res = []

    for c in c_list:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        
        res.append([x, y, d, w, h])
    
    return res

def bbb(img):
    '''
    bbb is short of body bound box
    :param data_dict:
    :return:
    '''
    boxs = []
    for i in range(img.shape[0]):
        if np.sum(img[i]) != 0:
            boxes = crop(img[i], i)
            for box in boxes:
                x, y, d, w, h = box
                x1, x2, y1, y2, d1, d2 = x, x+w, y, y+h, d, d
                for existed_box in boxs:
                    x0, x01, y0, y01, d0, _ = existed_box
                    w0 = x01 - x0
                    h0 = y01 - y0
                    if x0 > x - w0 and x0 < x + w and y0 > y - h0 and y0 < y + h:
                        boxs.remove(existed_box)
                        d1 = d0
                        if x0 < x1:
                            x1 = x0
                        if x0 + w0 > x2:
                            x2 = x0 + w0
                        if y0 < y1:
                            y1 = y0
                        if y0 + h0 > y2:
                            y2 = y0 + h0
                boxs.append([x1, x2, y1, y2, d1, d2])
    return boxs

train_list = []
for _, _, files in os.walk(os.path.join("data_crop", "train")):
    for f in files:
        id, _ = f.split(".")
        train_list.append(id.strip())
val_list = []
for _, _, files in os.walk(os.path.join("data_crop", "val")):
    for f in files:
        id, _ = f.split(".")
        val_list.append(id.strip())
train_list.sort()
val_list.sort()

with open("class_label.json", "r") as f:
    data = json.load(f)

label_list = ['异位甲状腺肿瘤', '胸腺癌', '纵隔软组织肿瘤', '胸腺瘤', '胸腺增生', '良性囊肿', '神经源性肿瘤', '淋巴瘤', '淋巴组织增生', '神经内分泌肿瘤', '纵隔生殖细胞瘤']
train_file = open("train.txt", "w")
val_file = open("val.txt", "w")

for id in train_list:
    idx = str(id).lstrip("0")
    try:
        label = data[idx]
    except:
        continue
    CT = np.load(os.path.join("data_crop", "train", f"{id}.npy"))
    seg = np.load(os.path.join("label_crop", "train", f"{id}.npy"))
    class_label = label_list.index(label)

    boxs = bbb(seg.astype(np.uint8))

    count = 1
    for b in boxs:
        x1, x2, y1, y2, d1, d2 = b
        np.save(os.path.join("class_data", f"{id}_{count}.npy"), CT[d1:d2+1, y1:y2+1, x1:x2+1])
        #np.save(os.path.join("class_data", f"{id}_{count}.npy"), CT[d1:d2+1])
        train_file.write(f"{id}_{count}.npy {class_label}\n")
        count += 1

for id in val_list:
    idx = str(id).lstrip("0")
    try:
        label = data[idx]
    except:
        continue
    CT = np.load(os.path.join("data_crop", "val", f"{id}.npy"))
    seg = np.load(os.path.join("label_crop", "val", f"{id}.npy"))
    class_label = label_list.index(label)

    boxs = bbb(seg.astype(np.uint8))

    count = 1
    for b in boxs:
        x1, x2, y1, y2, d1, d2 = b
        np.save(os.path.join("class_data", f"{id}_{count}.npy"), CT[d1:d2+1, y1:y2+1, x1:x2+1])
        #np.save(os.path.join("class_data", f"{id}_{count}.npy"), CT[d1:d2+1])
        val_file.write(f"{id}_{count}.npy {class_label}\n")
        count += 1