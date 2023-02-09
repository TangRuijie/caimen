# from pynvml import *
import copy
import importlib
import numpy as np

global params_list
global value_matrix
global stage_ind

params = {}
params_list = []
value_matrix = []
pre_items_dict = {}

stage_ind = 0
params['stage' + str(stage_ind)] = {}

# nvmlInit()
# deviceCount = nvmlDeviceGetCount()
# nvmlShutdown()
# init_av_gpus = ''
# for i in range(deviceCount):
#     init_av_gpus += str(i) + ','
# av_gpus = init_av_gpus[:-1]

def get_param_setting(set_name):
    params = importlib.import_module('setting.' + set_name)
    return params

def no_vis_show(params_list):
    show_num = 0
    for param in params_list:
        show_num += len(param.split())
    return show_num

def add_params(params_str, values, pre_items = []):

    def gen_allrank_tuples(value_data, tuple_list = [], cur_list = []):
        if len(value_data) == 0:
            tuple_list.append(tuple(cur_list))
        else:
            first_list = value_data[0]
            for v in first_list:
                tmp_list = copy.copy(cur_list)
                if isinstance(v, tuple):
                    tmp_list.extend(list(v))
                else:
                    tmp_list.append(v)
                gen_allrank_tuples(value_data[1:], tuple_list, tmp_list)

    def simplify_pre_list(pre_list):
        tmp_dict = {}
        for k, v in pre_list:
            if k not in tmp_dict.keys():
                tmp_dict[k] = {}
                tmp_dict[k]['v'] = []
                tmp_dict[k]['len'] = []
            tmp_dict[k]['v'].append(v)
            tmp_dict[k]['len'].append(len(v))

        new_list = []

        for k, data in tmp_dict.items():
            min_ind = np.argmin(data['len'])
            new_list.append([k, data['v'][min_ind]])

        return new_list

    if len(pre_items) == 0:
        params_list.append(params_str)
        value_matrix.append(values)
    else:
        new_str = params_str

        tmp_list = pre_items
        for item in pre_items:
            p_list = item[0].split()
            for p in p_list:
                if p in pre_items_dict.keys():
                    tmp_list.extend(pre_items_dict[p])

        pre_items = simplify_pre_list(tmp_list)

        p_list = params_str.split()
        for p in p_list:
            pre_items_dict[p] = pre_items

        for item in pre_items:
            new_str += ' ' + item[0]

        new_values = []
        value_data = [item[1] for item in pre_items]
        for v in values:
            p_num = len(params_str.split())
            if p_num == 1:
                v_tuple = [v]
            else:
                v_tuple = list(v)

            tuple_list = []
            gen_allrank_tuples(value_data, tuple_list, v_tuple)
            new_values.extend(tuple_list)

        params_list.append(new_str)
        value_matrix.append(new_values)

def upload_setting(vis_num = -1, memory = 10, av_gpus ='0', gpu_num = 1, start_time = 600):
    global params_list
    global value_matrix
    global stage_ind
    params['stage' + str(stage_ind)]['params'] = params_list
    params['stage' + str(stage_ind)]['values'] = value_matrix
    params['stage' + str(stage_ind)]['vis_num'] = vis_num
    params['stage' + str(stage_ind)]['memory'] = memory
    params['stage' + str(stage_ind)]['av_gpus'] = av_gpus
    params['stage' + str(stage_ind)]['gpu_num'] = gpu_num
    params['stage' + str(stage_ind)]['start_time'] = start_time

    params_list = []
    value_matrix = []
    stage_ind += 1

    params['stage' + str(stage_ind)] = {}