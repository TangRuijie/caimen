import copy
import time
import datetime
import json
import numpy as np
import setting
import random
import os
import argparse
from socket import *
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

server_ip = '127.0.0.1'
server_port = 6792
debug_valid_freq = 200
debug_niter = 1
debug_max_dataset_size = 100
debug_num_thread = 1
or_add_info = ''

gpu_list = ['0','1','2','3']
parser = argparse.ArgumentParser()
parser.add_argument('--manager', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--dnum', default=1, type = int)
parser.add_argument('--ex_file', default='', type = str)
parser.add_argument('--gpu_ids', default='', type = str)
parser.add_argument('--set_file', default='fpn_maskcomb', type = str)
parser.add_argument('--set_visible_device', default=True, type=bool)
parser.add_argument('--local_rank', type=int, default=0)
cfg = parser.parse_args()

def getTime():
    timeNow = datetime.datetime.now().strftime('%b%d_%H-%M')
    return timeNow

def split_to_list(total_num,split_num):
    base_num = total_num // split_num
    last_num = total_num % split_num

    cg_count = [base_num for _ in range(split_num)]
    for i in range(last_num):
        cg_count[i] += 1

    return cg_count

def value2str(v):
    if isinstance(v, list):
        v = json.dumps(v)

    v = str(v)
    v = v.replace(' ','')
    v = v.replace('\"','\\\"')
    return v

def generate_code(tmp_code_dict, tmp_text, p_count, code_dict_list, name_list, params_list,value_matrix):
    if len(params_list) == 0:
        new_code_dict = {}
        for k, v in tmp_code_dict.items():
            new_code_dict[k] = [v[0]]
        code_dict_list.append(new_code_dict)
        name_list.append(tmp_text)

    else:
        for value in value_matrix[0]:
            tmp_p_count = p_count
            tmp_params = params_list[0].split()
            if len(tmp_params) > 1:
                tmp_params = tmp_params[:len(value)]
                p_code_dict = copy.copy(tmp_code_dict)
                p_text = tmp_text

                param_match = True
                local_p_dict = {}
                local_p_text = ''
                for i in range(len(tmp_params)):
                    cur_param = tmp_params[i]
                    tmp_v = value2str(list(value)[i])

                    if cur_param not in p_code_dict.keys():
                        local_p_dict[cur_param] = []
                        local_p_dict[cur_param].append(tmp_v)
                    else:
                        if p_code_dict[cur_param][0] != tmp_v:
                            param_match = False
                            break

                if param_match:
                    for k, v_list in local_p_dict.items():
                        if k not in p_code_dict.keys():
                            p_code_dict[k] = []
                        p_code_dict[k].append(v_list[0])

                        if tmp_p_count < vis_num or vis_num < 0:
                            local_p_text += ',' + str(k) + '=' + str(p_code_dict[k][0])
                        tmp_p_count += 1
                    p_text += local_p_text

            else:
                p_code_dict = copy.copy(tmp_code_dict)
                p_text = tmp_text
                cur_param = params_list[0]
                if cur_param not in p_code_dict.keys():
                    p_code_dict[cur_param] = []
                p_code_dict[cur_param].append(value2str(value))
                if tmp_p_count < vis_num or vis_num < 0:
                    p_text += ',' + str(cur_param) + '=' + str(p_code_dict[cur_param][0])
                tmp_p_count += 1

            generate_code(p_code_dict, p_text, tmp_p_count, code_dict_list, name_list,params_list[1:],value_matrix[1:])

def dict2text(start_dict, code_dict_list):
    code_text_list = []
    for code_dict in code_dict_list:
        for k, v_list in start_dict.items():
            if k not in code_dict.keys():
                code_dict[k] = v_list

    for code_dict in code_dict_list:
        tmp_code = ''
        for k, v_list in code_dict.items():

            tmp_code += ' --' + str(k) + ' ' + str(v_list[0])

        code_text_list.append(tmp_code)

    return code_text_list

if __name__ == '__main__':

    start_code_dict = {}
    pre_check_dir = ''
    set_visible_device = True
    isTrain = not cfg.test
    isDebug = cfg.debug
    if isDebug:
        cfg.manager = False

    if cfg.gpu_ids != '':
        gpu_list = [cfg.gpu_ids]

    ex_file = cfg.ex_file
    if ex_file == '':
        if isTrain:
            ex_file = 'train.py'
        else:
            ex_file = 'test.py'

    if not isTrain:
        or_add_info = 'TEST_' + or_add_info

    p_code_num = 1 #task num for each gpus
    param_setting = setting.get_param_setting(cfg.set_file)
    c_time = getTime()
    ppid = c_time + '_' + str(os.getpid())

    for stage in range(len(param_setting.params.keys())):
        end_dir = 'buffer/training/' + ppid + '/' + str(stage) + '/'
        add_info = c_time + or_add_info
        add_info += 'stage'
        add_info += str(stage)

        code_dict_list = []
        name_list = []

        params_list = param_setting.params['stage' + str(stage)]['params']
        value_matrix = param_setting.params['stage' + str(stage)]['values']
        vis_num = param_setting.params['stage' + str(stage)]['vis_num']

        generate_code({},add_info,0,code_dict_list,name_list,params_list,value_matrix)
        code_list = dict2text(start_code_dict, code_dict_list)

        tmp_list1 = []
        tmp_list2 = []

        for c_code, c_name in zip(code_list, name_list):
            if c_code not in tmp_list1:
                tmp_list1.append(c_code)
                tmp_list2.append(c_name)
        code_list = tmp_list1
        name_list = tmp_list2

        print('set num ', len(code_list))
        print(code_list[0])

        if isDebug:
            code_list = code_list[:cfg.dnum]

        inds = list(range(len(code_list)))
        random.shuffle(inds)
        code_list = [code_list[ind] for ind in inds]
        name_list = [name_list[ind] for ind in inds]

        for i in range(len(code_list)):
            code_list[i] = ' python ' + str(ex_file) + ' ' + code_list[i]
            code_list[i] += ' --ppid ' + ppid
            code_list[i] += ' --pid ' + str(i)
            code_list[i] += ' --ad_stage ' + str(stage)
            if pre_check_dir != '':
                code_list[i] += ' --pre_check_dir ' + pre_check_dir
            if not isDebug:
                code_list[i] += ' --name ' + str(name_list[i])[:200].replace('/','_')
            else:
                code_list[i] += ' --name debug'

        if cfg.manager: # debug
            tcp_client_socket = socket(AF_INET, SOCK_STREAM)
            tcp_client_socket.connect((server_ip, server_port))
            av_gpus = param_setting.params['stage' + str(stage)]['av_gpus']
            gpu_num = param_setting.params['stage' + str(stage)]['gpu_num']
            memory = param_setting.params['stage' + str(stage)]['memory']
            start_time = param_setting.params['stage' + str(stage)]['start_time']
            meta_info = '(' + av_gpus + ' ' + str(gpu_num) + ' ' + str(memory) + ' ' + str(start_time) + ') '
            for i in range(len(code_list)):
                code_list[i] = meta_info + code_list[i]
                tcp_client_socket.send(code_list[i].encode('utf8'))
                tcp_client_socket.recv(1024)

            tcp_client_socket.send('finish'.encode('utf8'))
            tcp_client_socket.close()

            while True:
                if os.path.exists(os.path.join(end_dir, 'finish')):
                    print(cfg.set_file, 'Stage', stage, 'is finished.')
                    break
                else:
                    print(cfg.set_file, 'Stage', stage, 'is excuting.')
                time.sleep(60)


        else:
            gpu_num = min(len(gpu_list), len(code_list))
            cg_count = split_to_list(len(code_list), gpu_num)

            for i in range(len(cg_count)):
                cg_count[i] = split_to_list(cg_count[i],p_code_num)

            gpu_ind = 0
            code_count = 0
            bash = ''
            while gpu_ind < gpu_num:
                g_num_list = cg_count[gpu_ind]
                for g_num in g_num_list:
                    if g_num == 0:
                        continue
                    bash += '('
                    for i in range(g_num):
                        if set_visible_device:
                            bash += 'CUDA_VISIBLE_DEVICES=' + str(gpu_list[gpu_ind])
                        bash += code_list[code_count] + ',gpu_ids=' + str(gpu_list[gpu_ind])
                        tmp_gpus = gpu_list[gpu_ind]
                        if set_visible_device:
                            tmp_gpus = ''
                            tmp_num = len(gpu_list[gpu_ind].split(','))
                            for j in range(tmp_num):
                                tmp_gpus += str(j) + ','
                            tmp_gpus = tmp_gpus[:-1]

                        bash += ' --gpu_ids ' + tmp_gpus
                        if isDebug:
                            bash += ' --num_threads ' + str(debug_num_thread)
                            bash += ' --max_dataset_size ' + str(debug_max_dataset_size)
                            if isTrain:
                                bash += ' --valid_freq ' + str(debug_valid_freq)
                                bash += ' --niter ' + str(debug_niter)

                        bash += ';'
                        code_count += 1
                    bash += ')&'
                gpu_ind += 1
            bash = bash[:-1]
            os.system(bash)

        if isTrain:
            not_pid_list = ['finish', 'outf.txt', 'errf.txt']
            f_list = [f for f in os.listdir(end_dir) if f not in not_pid_list]
            pid_list = []
            value_list = []
            for fname in f_list:
                fname = os.path.join(end_dir,fname)
                with open(fname,'r') as f:
                    items = f.read().split()
                    pid = int(items[0])
                    save_dir = items[1]
                    value = float(items[-1])
                    pid_list.append(pid)
                    value_list.append(value)
                    ind = np.argmax(value_list).item()
                    max_pid = pid_list[ind]

            start_code_dict = code_dict_list[max_pid]
            pre_check_dir = save_dir
