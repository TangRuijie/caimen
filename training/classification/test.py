import os
from options.my_options import TestOptions
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer
import time
import os
import torch
import torch.distributed as distrib
torch.backends.cudnn.enabled = False
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

if __name__ == '__main__':
    distrib.init_process_group(backend="nccl")
    local_rank = distrib.get_rank()
    opt = TestOptions(local_rank).parse()  # get test options
    opt.serial_batches = True

    #if opt.l_state == 'train':
    #    opt.l_state = 'test'

    visualizer = Visualizer(opt)

    #if opt.test_train_data:
    #    v_dataset = create_dataset(opt, 'train')
    #else:
    v_dataset = create_dataset(opt, 'test')

    model = create_model(opt)
    model.setup(opt)

    v_start_time = time.time()
    model.validation(v_dataset, visualizer,valid_iter=-1)
    model.change_dir_name(visualizer)
