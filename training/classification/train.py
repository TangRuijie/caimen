from threading import local
from util.basic import *
from util.metrics import TimeMeter
from models import create_model
from options.my_options import TrainOptions
from datasets import create_dataset
from util.visualizer import Visualizer
import torch.distributed as distrib
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

if __name__ == '__main__':
    distrib.init_process_group(backend="nccl")
    local_rank = distrib.get_rank()
    opt = TrainOptions(local_rank).parse()   # get training options

    dataset = create_dataset(opt,'train')  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    if opt.valid_model:
        v_dataset = create_dataset(opt,'valid')
        print('The number of validation images = %d' % len(v_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.total_iters = 0                # the total number of training iterations
    opt.valid_iters = 0

    if dataset_size == 0:
        while True:
            model.average_weights(0)

    iter_time_meter = TimeMeter()
    data_time_meter = TimeMeter()
    epoch_time_meter = TimeMeter()

    if local_rank == 0:
        model.average_weights(1)
    else:
        model.average_weights(0)
    print('Start to train')

    batch_iter = 0
    early_stop = False
    epoch = -1
    while True:
        epoch += 1
    #for epoch in range(opt.epoch_count, 20):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        opt.epoch = epoch
        opt.epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_time_meter.start()  # timer for entire epoch
        data_time_meter.start()
        iter_time_meter.start()
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record()
            iter_time_meter.start()
            visualizer.reset()
            opt.total_iters += 1
            opt.epoch_iter += 1
            batch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if model.wait_over and local_rank == 0:
                early_stop = True
            #    early_stop = model.check_early_stop(1)
            #else:
            #    early_stop = model.check_early_stop(0)
            #model.check_finish(0)
            model.average_weights(dataset_size)
            #model.average_weights(1)
            iter_time_meter.record()

            visualizer.plot_current_info(model, opt.total_iters, ptype='train')

            if opt.total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                visualizer.print_current_info(epoch, opt.epoch_iter, model, iter_time_meter.val, data_time_meter.val)

            if opt.total_iters % opt.valid_freq == 0:
                model.update_metrics('global')
                model.save_networks('train', visualizer)
                visualizer.print_global_info(epoch, opt.epoch_iter, model, iter_time_meter.sum/60,data_time_meter.sum/60)
                visualizer.plot_global_info(model, opt.valid_iters, ptype='train')

                iter_time_meter.reset()
                data_time_meter.reset()

                model.reset_meters()
                model.clear_info()
                if opt.valid_model and local_rank == 0:
                    model.validation(v_dataset, visualizer, valid_iter= opt.valid_iters)
                model.update_learning_rate()

                save_suffix = 'optimal'
                model.save_networks(save_suffix,visualizer)

                if early_stop:
                    break

                model.reset_meters()
                opt.valid_iters += 1

            data_time_meter.start()
            iter_time_meter.start()

        if early_stop:
            print('early stop at %d / %d' % (epoch,opt.epoch_iter))
            break

        epoch_time_meter.record()
        epoch_time_meter.start()

        #while True:
        #    if not model.check_finish(1):
        #        model.average_weights(0)
        #    else:
        #        break
        
        dataset.dataset.next_epoch()
        model.next_epoch()

        print('End of epoch %d / %d \t Time Taken: %d hours' % (epoch, opt.niter + opt.niter_decay, epoch_time_meter.sum/3600.))

        if local_rank == 0 and epoch == 30:
            break
    v_dataset = create_dataset(opt, 'test')
    model.validation(v_dataset, visualizer, valid_iter=-2)
    model.change_dir_name(visualizer, change_type='add')