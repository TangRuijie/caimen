import os.path as osp
import numpy as np
from util.basic import *
from datetime import datetime


def move_to_his(target_dir, day_thred = 10):
    nowday = datetime.now().day
    nowyear = datetime.now().year

    check_dir = 'checkpoints'
    run_dir = 'runs'

    target_check_dir = osp.join(target_dir, check_dir, str(nowyear))
    target_run_dir = osp.join(target_dir, run_dir, str(nowyear))

    mkdir(target_run_dir)
    mkdir(target_check_dir)

    target_dir_list = [target_check_dir, target_run_dir]
    for i, mv_dir in enumerate([check_dir, run_dir]):
        dir_list = os.listdir(mv_dir)
        for d in dir_list:
            try:
                check_time = d.split('_')[0]
                check_time = datetime.strptime(check_time, '%b%d')
            except:
                continue

            if (nowday - check_time.day) % 366 > day_thred:
                cmd = 'mv ' + osp.join(mv_dir, d) + ' ' + target_dir_list[i]
                os.system(cmd)


def txt2json(check_dir):
    if osp.isdir(check_dir):
        if osp.exists(osp.join(check_dir, 'optimal_pred_result.txt')):
            fpath = osp.join(check_dir, 'optimal_pred_result.txt')
        else:
            fpath = osp.join(check_dir, 'pred_result.txt')
    else:
        fpath = check_dir

    with open(fpath) as f:
        lines = f.readlines()
    log_data = [[], [], []]

    for line in lines:
        line = line.rstrip()
        items = line.split()
        log_data[0].append(items[-1])
        log_data[1].append(int(items[1]))
        log_data[2].append(float(items[2]))

    return log_data

def clean_data(y_trues, y_scores, y_paths = [], data_num = 10, reverse = False):
    inds_0_list = []
    inds_1_list = []
    bias_0_list = []
    bias_1_list = []
    for i, label in enumerate(y_trues):
        if label == 1:
            tmp_score = 1 - y_scores[i]
            if reverse:
                tmp_score = 1 - tmp_score
            bias_1_list.append(tmp_score)
            inds_1_list.append(i)
        if label == 0:
            tmp_score = y_scores[i]
            if reverse:
                tmp_score = 1 - tmp_score
            bias_0_list.append(tmp_score)
            inds_0_list.append(i)

    sort_0_inds = np.argsort(bias_0_list).tolist()
    sort_1_inds = np.argsort(bias_1_list).tolist()

    max_0_inds = sort_0_inds[-data_num:]
    max_1_inds = sort_1_inds[-data_num:]

    max_inds = []
    for m_ind in max_0_inds:
        max_inds.append(inds_0_list[m_ind])
    for m_ind in max_1_inds:
        max_inds.append(inds_1_list[m_ind])

    max_inds = sorted(max_inds, reverse=True)

    rm_y_paths = []
    rm_y_trues = []
    rm_y_scores = []


    for m_ind in max_inds:
        # if y_trues[m_ind] == 1:
        #     print(m_ind)
        #     print(y_trues[m_ind])
        #     print(y_scores[m_ind])
        #     print()
        #

        rm_y_trues.append(y_trues[m_ind])
        rm_y_scores.append(y_scores[m_ind])
        del y_trues[m_ind]
        del y_scores[m_ind]

        if len(y_paths):
            rm_y_paths.append(y_paths[m_ind])
            del y_paths[m_ind]

    if len(y_paths):
        return rm_y_paths, rm_y_trues, rm_y_scores
    else:
        return rm_y_trues, rm_y_scores


