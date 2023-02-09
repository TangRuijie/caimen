import os
import json
import numpy as np
import os.path as osp
import sys
sys.path.insert(0, '.')
from util.basic import read_multi_data

data_root = '/GPUFS/gyfyy_jxhe_1/User/tangruijie/DATA/Classification/data'

label_list = ['胸腺瘤', '良性囊肿', '神经源性肿瘤', '胸腺癌', '纵隔生殖细胞瘤', '纵隔软组织肿瘤', '淋巴瘤', '神经内分泌肿瘤', '淋巴组织增生', '异位甲状腺肿瘤', '胸腺增生', '肉芽肿性炎']
def mk_data_json(split_ratio = 0.2):
    test_name = ['中山','青岛','西安']
    d_list = os.listdir(data_root)
    train_d_list = []
    test_d_list = []
    for d in d_list:
        in_test = False
        for t_name in test_name:
            if d.startswith(t_name):
                in_test = True
        if not in_test:
            train_d_list.append(d)
        else:
            test_d_list.append(d)

    'train d'
    print(train_d_list)
    'test d'
    print(test_d_list)

    p_list = [osp.join(data_root, d) for d in train_d_list]
    test_p_list = [osp.join(data_root, d) for d in test_d_list]

    train_dict = {}
    val_dict = {}

    for p in p_list:
        print(p)
        data_dict = {}
        np_path_list = [osp.join(p, np) for np in os.listdir(p) if np.endswith('.npz')]
        info_list = read_multi_data(read_cls, np_path_list, workers=50)
        print('data num',len(info_list))
        for path, label in info_list:
            if label not in data_dict.keys():
                data_dict[label] = []
            data_dict[label].append(path)

        for label, path_list in data_dict.items():
            # print('label',label,len(path_list))

            print(label_list[label],':',len(path_list))
            if label not in train_dict.keys():
                train_dict[label] = []
                val_dict[label] = []

            val_num = min(int(len(path_list) * split_ratio), 20)

            train_dict[label].extend(path_list[val_num:])
            val_dict[label].extend(path_list[:val_num])

    print('train num')
    train_num = 0


    for label in range(0,12):
        path_list = train_dict[label]
        print(label_list[label],':',len(path_list))
        train_num += len(path_list)
    print('total', train_num)

    print('val num')
    val_num = 0
    for label in range(0,12):
        path_list = val_dict[label]
        # for label, path_list in val_dict.items():
        print(label_list[label],':',len(path_list))
        # print('label:',label,len(path_list))
        val_num += len(path_list)
    print('total', val_num)

    with open('buffer/full_data_train.json','w') as f:
        json.dump(train_dict, f)
    with open('buffer/full_data_valid.json','w') as f:
        json.dump(val_dict, f)


    test_dict = {}
    for p in test_p_list:
        print(p)
        np_path_list = [osp.join(p, np) for np in os.listdir(p) if np.endswith('.npz')]
        info_list = read_multi_data(read_cls, np_path_list, workers=50)
        print('data num',len(info_list))
        for path, label in info_list:
            if label not in test_dict.keys():
                test_dict[label] = []
            test_dict[label].append(path)
    with open('buffer/full_data_test.json','w') as f:
        json.dump(test_dict, f)


def read_cls(np_path):
    cls = np.load(np_path)['cls'].item()
    return np_path, cls

if __name__ == '__main__':
    mk_data_json()
