import re
import os
import time
import _thread
import subprocess
from datetime import datetime
from util.log import move_to_his
from pynvml import *
from socket import *
from util.basic import *
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
set_visible_device = True

address = ('', 6792)
his_dir = '/home2'

gpu_dict = {}
allocated_gpu_set = set()

kill_wait_dict = {}
kill_wait_iter = 100

process_dict = {}
wait_process_list = []
gpu_start_set = set()

def get_gpu_mem(d):
    handle = nvmlDeviceGetHandleByIndex(d)
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    free_m = meminfo.free / 1024 / 1024 / 1024
    return free_m

    # total_m = meminfo.total
    # free_m = meminfo.free
    # if free_m / total_m > 0.8:
    #     return True
    # else:
    #     return False
    # if free_m < p_mems[i]:
    #     print('Waiting to run on devices ' + devices)
    #     is_free = False
    #     w_count = 0
    #     break

def del_info_in_cmd(cmd, key = 'ppid'):
    print('cmd ', cmd)
    if '--' + key not in cmd:
        return cmd

    ppid_loc = cmd.index('--' + key)
    sub_str = cmd[ppid_loc + len('--' + key + ' '):]
    if ' ' not in sub_str:
        return cmd[:ppid_loc]
    else:
        em = sub_str.index(' ') + ppid_loc + len('--' + key + ' ')
        cmd = cmd[:ppid_loc] + cmd[em + 1:]
    return cmd

def ex_info_from_cmd(cmd, key = 'ppid'):
    if '--' + key not in cmd:
        return None

    ppid_loc = cmd.index('--' + key)
    sub_str = cmd[ppid_loc + len('--' + key + ' '):]
    if ' ' not in sub_str:
        ppid = sub_str.rstrip()
    else:
        em = sub_str.index(' ') + ppid_loc + len('--' + key + ' ')
        ppid = cmd[ppid_loc + len('--' + key + ' '): em]
    return ppid

def replace_info_in_cmd(cmd, key, value):
    if '--' + key not in cmd:
        new_str = cmd + ' --' + key + ' ' + str(value)
        return new_str

    ppid_loc = cmd.index('--' + key)
    sub_str = cmd[ppid_loc + len('--' + key + ' '):]

    if ' ' not in sub_str:
        new_str = cmd[:ppid_loc]
    else:
        em = sub_str.index(' ') + ppid_loc + len('--' + key + ' ')
        new_str = cmd[:ppid_loc] + cmd[em + 1:]

    new_str = new_str[:ppid_loc] + '--' + key + ' ' + str(value) + ' ' + new_str[ppid_loc:]
    return new_str

def ex_meta_info(s):
    lk = s.index('(')
    rk = s.index(')')
    meta_info = s[lk + 1: rk]
    g_list, use_num, mem_size, start_time = meta_info.split()
    g_list = [int(g) for g in g_list.split(',')]
    return g_list, int(use_num), float(mem_size), int(start_time)

def excutable_gpu(cmd, mem_dict):
    global gpu_dict
    ppid = ex_info_from_cmd(cmd)
    av_gpu_list = process_dict[ppid]['av_gpu']
    use_num = process_dict[ppid]['use_gpu_num']
    need_mem = process_dict[ppid]['mem_size']

    tmp_list = []
    for g in av_gpu_list:
        if mem_dict[g] >= need_mem and g not in gpu_start_set and int(g) in allocated_gpu_set:
            tmp_list.append(g)
            if len(tmp_list) >= use_num:
                return tmp_list
        else:
            if len(gpu_dict[g]) == 0:
                kill_wait_dict[g] += 1

                if kill_wait_dict[g] >= kill_wait_iter:
                    pass
                    # os.system('python tools/kill_nvidia.py -d ' + str(g))
                    kill_wait_dict[g] = 0

    return None

def receive_process():
    global wait_process_list
    global allocated_gpu_set
    while True:
        client_socket, clientAddr = tcp_server_socket.accept()

        tmp_list = []
        while True:
            recv_data = client_socket.recv(1024)
            r_data = recv_data.decode('utf8')

            if r_data == 'finish':
                client_socket.close()
                break

            if r_data.lower().startswith('allocated_gpus'):
                print('the old allocated gpu set ', allocated_gpu_set)
                tmp_devices = r_data.rstrip().split()[1]
                tmp_devices = tmp_devices.split(',')
                allocated_gpu_set = set([int(d) for d in tmp_devices])
                print('the new allocated gpu set ', allocated_gpu_set)
                break

            ppid = ex_info_from_cmd(r_data)
            g_list, use_gpu_num, mem_size, start_time = ex_meta_info(r_data)
            cmd = r_data.split(')')[1]

            if ppid not in process_dict.keys():
                process_dict[ppid] = {}
                process_dict[ppid]['num'] = 1
                process_dict[ppid]['av_gpu'] = g_list
                process_dict[ppid]['mem_size'] = mem_size
                process_dict[ppid]['use_gpu_num'] = use_gpu_num
                process_dict[ppid]['start_time'] = start_time
                stage = ex_info_from_cmd(cmd, 'ad_stage')
                process_dict[ppid]['end_dir'] = 'buffer/' + ppid + '/' + str(stage) + '/'
                mkdir(process_dict[ppid]['end_dir'])

            else:
                process_dict[ppid]['num'] += 1

            tmp_list.append(cmd)
            print('receive a new procedure from', ppid)
            client_socket.send('receive'.encode('utf8'))

        if len(tmp_list):
            wait_process_list = tmp_list + wait_process_list
        time.sleep(0.1)


def count_time(ds, cost_time):
    time.sleep(cost_time)
    for d in ds:
        gpu_start_set.remove(d)

def refine_cmd(cmd):
    '''remove cuda'''
    p_loc = cmd.index('python')
    cmd = cmd[p_loc:]
    cmd = del_info_in_cmd(cmd, key='gpu_ids')
    name = ex_info_from_cmd(cmd, 'name')
    if 'gpu_ids=' in name:
        tmp_ind = name.index('gpu_ids=')
        name = name[:tmp_ind]
    cmd = del_info_in_cmd(cmd, key='name')
    cmd += ' --name ' + name.rstrip()
    return cmd

def ex_cmd(cmd, gpu_to_use):
    global process_dict

    time.sleep(30)

    name_wrong = False
    name = ex_info_from_cmd(cmd, 'name')
    ppid = ex_info_from_cmd(cmd)
    finish = False
    try:
        check_dir = os.path.join('checkpoints', name)
        mkdir(check_dir)
        outf_path = os.path.join(check_dir, 'outf.txt')
        errf_path = os.path.join(check_dir, 'errf.txt')
        outf = open(outf_path, 'w')
        errf = open(errf_path, 'w')
    except:
        print('some error occurs in name format')
        name_wrong = True

    if not name_wrong:
        p = subprocess.Popen(cmd, shell=True, stdout=outf, stderr=errf)
        while p.poll() is None:
            time.sleep(1)

        outf.close()
        errf.close()

        cmd = is_common_error(cmd)
        if cmd == '':
            finish = True

        if not finish:
            process_dict[ppid]['num'] -= 1
            for g in gpu_to_use:
                gpu_dict[g].remove(ppid)
            cmd = refine_cmd(cmd)
            wait_process_list.append(cmd)
            return

    process_dict[ppid]['num'] -= 1
    for g in gpu_to_use:
        gpu_dict[g].remove(ppid)

    if process_dict[ppid]['num'] == 0:
        finish_p = os.path.join(process_dict[ppid]['end_dir'], 'finish')
        with open(finish_p, 'w') as f:
            f.write('finish')
        process_dict.pop(ppid)

def is_common_error(cmd):
    name = ex_info_from_cmd(cmd, 'name')
    check_list = os.listdir('checkpoints')
    check_path = ''
    for check in check_list:
        if check.startswith(name):
            check_path = osp.join('checkpoints', check)
            break

    if osp.exists(osp.join(check_path, 'finish')):
        return ''

    err_path = osp.join(check_path, 'errf.txt')
    with open(err_path, 'r') as f:
        err = f.read().lower()

    print('error')
    print(cmd)
    print(err)
    error_dict = {}
    error_dict['num_threads'] = ['insufficient shared memory', 'segmentation fault', 'out of memory', 'runtimeerror: dataloader worker']

    def get_error_type(err):
        for k, v_list in error_dict.items():
            for v in v_list:
                if v in err:
                    return k

        return None

    error_type = get_error_type(err)
    if error_type is None:
        stop_path = osp.join(check_path, 'stop')
        with open(stop_path, 'w') as f:
            f.write('1')
        return ''

    if error_type == 'num_threads':
        num_thread = ex_info_from_cmd(cmd, 'num_threads')
        if num_thread is not None:
            new_thread = int(num_thread) // 2
        else:
            new_thread = 5
        cmd = replace_info_in_cmd(cmd, 'num_threads', new_thread)

    '''replace time'''

    timeNow = datetime.now().strftime('%b%d_%H-%M')

    old_name = ex_info_from_cmd(cmd, 'name')
    new_name = timeNow + old_name[len(timeNow):]
    cmd = replace_info_in_cmd(cmd, 'name',new_name)

    f_list = [f for f in os.listdir(check_path) if f.endswith('.pth')]
    if len(f_list) > 0:
        cmd = replace_info_in_cmd(cmd, 'load_dir', check_path)

    print('Restart')
    print(cmd)
    return cmd


def ex_process():
    global wait_process_list
    global gpu_start_set
    while True:
        free_mem_dict = {}
        for g in gpu_dict.keys():
            free_mem = get_gpu_mem(g)
            free_mem_dict[g] = free_mem

        i = 0
        while i < len(wait_process_list):
            gpus_to_use = excutable_gpu(wait_process_list[i], mem_dict = free_mem_dict)
            if gpus_to_use is None:
                i += 1
                continue

            cmd = wait_process_list[i]
            wait_process_list = wait_process_list[:i] + wait_process_list[i + 1:]

            ppid = ex_info_from_cmd(cmd)
            for g in gpus_to_use:
                gpu_dict[g].append(ppid)

            tmp_gpus = ''
            for g in gpus_to_use:
                tmp_gpus += str(g) + ','
            tmp_gpus = tmp_gpus[:-1]

            cmd += ',gpu_ids='+str(tmp_gpus)

            if set_visible_device:
                tmp_str = 'CUDA_VISIBLE_DEVICES='
                cmd = tmp_str + tmp_gpus + ' ' + cmd

                p_gpus = ''
                for j in range(len(gpus_to_use)):
                    p_gpus += str(j) + ','
                p_gpus = p_gpus[:-1]
            else:
                p_gpus = tmp_gpus

            cmd += ' --gpu_ids ' + p_gpus
            _thread.start_new_thread(ex_cmd, (cmd, gpus_to_use,))
            gpu_start_set = gpu_start_set | set(gpus_to_use)
            _thread.start_new_thread(count_time, (gpus_to_use, process_dict[ppid]['start_time'],))
        time.sleep(10)

def manager_process():
    global gpu_dict
    global his_dir
    has_moved = False
    print_wait = 0
    print_info = 0
    l_print_info = ''
    while True:
        '''Process info'''
        print_info = ''
        if len(wait_process_list) > 0:
            print_info += 'There are ' + str(len(wait_process_list)) + ' processes to be excuted. \n'
        else:
            print_info += 'No process is waiting. \n'

        ex_num = 0
        if len(gpu_dict.keys()) > 0:
            for k, v in gpu_dict.items():
                if len(v) > 0:
                    ex_num += len(v)
                    # print('gpu',k, ': ' + ', '.join(v))

        if not ex_num:
            print_info += 'No process is excuting. \n'
        else:
            print_info += str(ex_num) + ' process is excuting. \n'

        '''Check datetime info'''

        nowtime = datetime.now()
        if nowtime.day % 10 == 8 and not has_moved:
            move_to_his(target_dir=his_dir)
            has_moved = True
        else:
            has_moved = False

        if print_info != l_print_info or print_wait >= 12:
            print(print_info)
            l_print_info = print_info
            print_wait = 0
        else:
            print_wait += 1

        time.sleep(5)

if __name__ == '__main__':
    tcp_server_socket = socket(AF_INET, SOCK_STREAM)
    tcp_server_socket.bind(address)
    tcp_server_socket.listen(128)
    print('The port is ready')
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    allocated_gpu_set = set(list(range(deviceCount)))
    for i in range(deviceCount):
        gpu_dict[i] = []
        kill_wait_dict[i] = 0
    _thread.start_new_thread(receive_process, ())
    _thread.start_new_thread(ex_process, ())
    _thread.start_new_thread(manager_process,())

    while True:
        pass
