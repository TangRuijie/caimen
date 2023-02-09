"""This module contains simple helper functions """
from __future__ import print_function
import torch.multiprocessing as multiprocessing
import torch
import signal
import json
import numpy as np
from PIL import Image
from functools import wraps
import os
import zipfile
import tqdm
import time
import os.path as osp

global_c_time = 0

def read_multi_data(func, param_list, workers = 3, ignore_none = True, vis_schedule = False, ignore_bug = True):
    param_data = [[] for _ in range(workers)]
    c_worker = 0
    for i, param in enumerate(param_list):
        param_data[c_worker].append((i,param))
        c_worker = (c_worker + 1) % workers

    q = multiprocessing.Queue()
    q.cancel_join_thread()

    count = 0

    def read_data(func, param_part_list):

        if len(param_part_list) > 100 or vis_schedule:
            param_part_list = tqdm.tqdm(param_part_list)
        for i, param in param_part_list:
            if ignore_bug:
                try:
                    if isinstance(param,list) and func.__code__.co_argcount > 1:
                        data = func(*param)
                    else:
                        data = func(param)
                except:
                    data = None
            else:
                if isinstance(param,list) and func.__code__.co_argcount > 1:
                    data = func(*param)
                else:
                    data = func(param)


            q.put((i, data))

    for i in range(workers):
        w = multiprocessing.Process(
            target=read_data,
            args=(func, param_data[i]))
        w.daemon = False
        w.start()

    data_list = [None for _ in range(len(param_list))]

    while count < len(param_list):
        i, data = q.get()
        data_list[i] = data
        count += 1

    new_data_list = []
    if ignore_none:
        for data in data_list:
            if data is not None:
                new_data_list.append(data)
        data_list = new_data_list

    return data_list

def numpy2table(cmatrix):
    table = '| |'
    for i in range(cmatrix.shape[0]):
        table += str(i) + '|'
    table += '\n|:-:|'
    for j in range(cmatrix.shape[0]):
        table += ':-:|'
    table += '\n'

    for i in range(cmatrix.shape[0]):
        table += '|' + str(i) + '|'
        for j in range(cmatrix.shape[0]):
            table += str(cmatrix[i,j]) + '|'
        table += '\n'

    return table

# def tensor2im(input_image, imtype=np.uint8):
#     """"Converts a Tensor array into a numpy image array.
#
#     Parameters:
#         input_image (tensor) --  the input image tensor array
#         imtype (type)        --  the desired type of the converted numpy array
#     """
#     if not isinstance(input_image, np.ndarray):
#         if isinstance(input_image, torch.Tensor):  # get the data from a variable
#             image_tensor = input_image.data
#         else:
#             return input_image
#         image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
#         if image_numpy.shape[0] == 1:  # grayscale to RGB
#             image_numpy = np.tile(image_numpy, (3, 1, 1))
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
#     else:  # if it is a numpy array, do nothing
#         image_numpy = input_image
#     return image_numpy.astype(imtype)

def str2json(s):
    if isinstance(s, str):
        s = json.loads(s)
    return s

def dismantle_tuples(tuple_inputs):
    if not isinstance(tuple_inputs, list):
        return tuple_inputs
    new_list = []
    for t in tuple_inputs:
        new_list.append(t[0])
    return new_list

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """

    dirs = path.split('/')
    tmp_path = ''
    for d in dirs:
        tmp_path += d

        if len(d) > 0 and not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        tmp_path += '/'

def save_code(dir_path, zip_path):
    '''
    :param dir_path: 目标文件夹路径
    :param zip_path: 压缩后的文件夹路径
    :return:
    '''

    zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    for root, dirnames, filenames in os.walk(dir_path):
        file_path = root.replace(dir_path, '')
        for filename in filenames:
            if filename.endswith('.py'):
                zip.write(os.path.join(root, filename), os.path.join(file_path, filename))
    zip.close()

def save_experiment_buffer(opt):
    end_dir = 'buffer/' + opt.ppid + '/' + opt.ad_stage + '/'
    mkdir(end_dir)
    end_fname = end_dir + 'result_' + opt.name + '_' + str(opt.best_m_value)
    with open(end_fname, 'w') as f:
        f.write(opt.pid + ' ' + opt.save_dir + ' ' + str(opt.best_m_value))

def within_limit_time(func):
    def handler(signum, frame):
        raise Exception("the time is out of range")

    @wraps(func)
    def time_decorator(*args, **kwargs):
        try:
            signal.signal(signal.SIGALRM, handler=handler)
            if 'signal_time' in kwargs.keys():
                signal.alarm(kwargs['signal_time'])
            else:
                signal.alarm(600)

            s = func(*args, **kwargs)
            return s
        except:
            print('over time')
            return None

    return time_decorator

def stat_ex_time(c_step = 0):
    c_step = str(c_step)
    global global_c_time
    if c_step == '0':
        global_c_time = time.time()
    else:
        print('time step', c_step, ' ', time.time() - global_c_time)
        global_c_time = time.time()
