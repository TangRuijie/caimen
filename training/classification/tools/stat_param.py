import os
import numpy as np

def ex_valid_metric(path):

    s = path.split('_')[-1]
    if 'ids=' not in s:
        return float(s)
    else:
        return None

if __name__ == '__main__':
    param_list = [
        'embd_pos=0',
        'embd_pos=1',
    ]

    path_list = os.listdir('checkpoints')

    for param in param_list:
        tmp_list = list(filter(lambda x: param in x, path_list))
        v_list = [ex_valid_metric(p) for p in tmp_list]
        v_list = list(filter(lambda x: x is not None, v_list))
        print(param, np.mean(v_list))
