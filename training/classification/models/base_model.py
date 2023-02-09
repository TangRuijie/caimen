from util import *
import json
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import torch.nn as nn
import time
import copy
import util.metrics as metrics
from util.optimizer.adabound import AdaBound
from util.model import combine_with_id
from schedulers import get_scheduler
from schedulers import metric_schedulers
import torch.nn.functional as f


class BaseModel(object):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.wait_over = False
        self.start_forward = True
        self.wait_epoch = 0
        self.best_top1_acc = -1
        self.best_top3_acc = -1
        self.c_grad_iter = 0
        self.gpu_ids = opt.gpu_ids
        self.o_save_dir = opt.save_dir
        self.save_dir = opt.save_dir
        self.vis_dir = os.path.join(opt.vis_dir, opt.name)

        '''The buffer data to store necessary results'''
        self.buffer_ginput_ids = []
        self.buffer_gscores = []
        self.buffer_glabels = []
        self.buffer_gpreds = []

        '''The parameters to be reset for a new model'''

        self.loss_names = ['c'] # used to update networks,
        self.s_metric_names = ['accuracy_1'] # scalar metric, stat local infomation
        self.g_metric_names = [] # scalar metric, stat global infomation
        self.t_metric_names = ['cmatrix'] # table or matrix metric
        self.net_names = []

        self.valid_metric = 'accuracy_1'
        self.scheduler_metric = 'accuracy_1'

    @staticmethod
    def modify_commandline_options(parser):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        return parser

    @staticmethod
    def supply_option_info(opt):
        return opt

    def gen_meters(self):
        name_types = ['loss','s_metric','t_metric','g_metric']
        meters = {}
        for ntype in name_types:
            name_list = getattr(self,ntype + '_names')
            for name in name_list:
                meters[name] = metrics.Meter()
        return meters

    def update_metrics(self,m_type = 'local'):
        if not self.start_forward and m_type != 'global':
            return

        if m_type == 'global':
            name_types = ['t_metric', 'g_metric']
        else:
            name_types = ['loss', 's_metric']

        for ntype in name_types:
            cal_func = getattr(self,'cal_' + ntype)
            cal_func()

            name_list = getattr(self,ntype + '_names')

            for name in name_list:
                try:
                    self.update_meters(ntype,name)
                except:
                    if not hasattr(self, ntype + '_' + name):
                        raise ValueError(ntype + '_' + name + ' does not exist')
                    else:
                        value = getattr(self, ntype + '_' + name)
                        raise ValueError('the value of ' + ntype + '_' + name + ': ' + str(value))

    def average_weights(self, dataset_size):
        pass

    def check_early_stop(self, es_signal):
        pass

    def check_finish(self, finish_signal):
        pass

    def average_gradients(self):
        pass

    def update_meters(self,ntype,name):
        value = getattr(self, ntype + '_' + name)

        if isinstance(value,torch.Tensor):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray) and ntype != 't_metric':
            value = value.item()

        if ntype != 't_metric':
            self.meters[name].update(value,self.input_size)
        else:
            self.meters[name].update(value,1)

    def reset_meters(self):
        name_types = ['loss', 's_metric', 't_metric', 'g_metric']

        for ntype in name_types:
            name_list = getattr(self, ntype + '_names')
            for name in name_list:
                # value = getattr(self, ntype + '_' + name)
                self.meters[name].reset()

    # @abstractmethod
    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        self.input = data[0]
        self.label = data[1]
        self.input_id = data[2]
        self.input = self.input.cuda(self.gpu_ids[0])
        self.label = self.label.cuda(self.gpu_ids[0])
        self.input_size = self.input.shape[0]

    def get_parameters(self):
        names = self.net_names
        p_list = []
        for name in names:
            net = getattr(self, 'net_' + name)
            p_list.append(net.parameters())

        if len(p_list) == 1:
            return p_list[0]
        else:
            n_p_list = []
            for p in p_list:
                tmp_p = {}
                tmp_p['params'] = p
                n_p_list.append(tmp_p)
            return n_p_list

    def clear_info(self):
        # print(' buffer name ' + str(self.buffer_names))
        for name in self.buffer_names:
            if name == 'names':
                continue
            # tmp_buffer = getattr(self,'buffer_' + name)
            # if len(tmp_buffer) > 0:
            #     if isinstance(tmp_buffer[0],list):
            #         tmp_buffer = [[] for _ in range(len(tmp_buffer))]
            #     else:
            #         tmp_buffer = []
            tmp_buffer = []
            setattr(self,'buffer_' + name,tmp_buffer)

        # self.detach_from_gpu()

    def set_optimizer(self,opt):
        params = self.get_parameters()
        if not len(params):
            self.optimizers = []
            return
        if opt.op_name == 'SGD':
            optimizer = torch.optim.SGD(params, lr=opt.lr, momentum = opt.momentum, nesterov=opt.nesterov,
                                             weight_decay = opt.weight_decay)
        elif opt.op_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = opt.weight_decay)

        elif opt.op_name.lower() == 'adabound':
            print('The optimizer is adabound')
            optimizer = AdaBound(params, lr=opt.lr, final_lr=opt.final_lr)
            
        self.optimizers = [optimizer]

    @abstractmethod
    def forward(self):
        net = getattr(self, 'net_' + self.net_names[0])
        self.y = net(self.input)
        self.score, _ = f.softmax(torch.max(self.y.detach(), dim = 1))

    def validate_parameters(self):
        if not self.start_forward:
            return
        if self.opt.vis_method != 'gradcam':
            with torch.no_grad():
                self.forward()
        else:
            self.forward()
        self.stat_info()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        if not self.start_forward:
            return
        self.forward()
        self.backward()
        
        self.stat_info()
        self.c_grad_iter += 1

        if self.c_grad_iter == self.opt.grad_iter_size:
            self.clip_grad()
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
            self.c_grad_iter = 0

    def stat_info(self):
        self.label = self.label.cpu().long()
        self.y = self.y.cpu()
        self.score = self.score.cpu()

        if len(self.label.shape) == 1:
            if self.y.shape[1] > 1:
                pred = torch.argmax(self.y, dim=1)
            else:
                pred = (self.y.view(-1) > self.opt.recall_thred).long()
        else:
            pred = (self.y > self.opt.recall_thred).long()

        self.pred = pred

        self.buffer_gscores.extend(self.score.tolist())
        self.buffer_glabels.extend(self.label.tolist())
        self.buffer_gpreds.extend(self.pred.tolist())
        self.buffer_ginput_ids.extend(self.input_id)

        self.visualize()

    def visualize(self):
        pass

    def get_buffer_names(self):
        v_names = list(self.__dict__.keys())
        b_names = [v.replace('buffer_','') for v in v_names if v.startswith('buffer')]
        # print(b_names)
        return b_names

    def zero_grad(self):
        for n_name in self.net_names:
            net = getattr(self,'net_' + n_name)
            net.zero_grad()

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.buffer_names = self.get_buffer_names()
        self.meters = self.gen_meters()
        self.schedulers = []

        # if self.opt.l_state == 'train':
        #     self.set_train_load_dir()
        # else:
        #     self.set_valid_load_dir()
        #
        if opt.load_dir != '' and opt.load_net:
            load_suffix = 'optimal'
            self.load_networks(load_suffix, load_dir=opt.load_dir, strict=self.opt.load_strict)
            self.print_networks(opt.verbose)

        if opt.l_state == 'train':
            self.set_optimizer(opt)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # else:
        #     load_suffix = 'optimal'
        #     self.load_networks(load_suffix)
        #     self.print_networks(opt.verbose)

        if self.opt.save_base_model:
            for name in self.net_names:
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % ('base', name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    net = getattr(self, 'net_' + name)
                    torch.save(net.state_dict(), save_path)

        net_names = self.net_names

        # visualization
        if opt.vis_method is not None and opt.vis_method != '':
            self.vis_method = vis_method_dict[opt.vis_method](self)

        for name in net_names:
            net = getattr(self,'net_' + name)
            net.cuda(self.gpu_ids[0])
            if len(self.gpu_ids) > 1:
                setattr(self,'net_' + name,nn.DataParallel(net,opt.gpu_ids))

        if self.opt.l_state == 'train':
            self.train()
        else:
            self.eval()

    def cal_loss(self):
        loss_list = []
        self.loss_c = nn.CrossEntropyLoss()(self.y, self.label)
        loss_list.append(self.loss_c)
        return loss_list

    def cal_s_metric(self):
        acc_top_list = []
        for m_name in self.s_metric_names:
            if m_name.startswith('accuracy'):
                num = int(m_name.split('_')[-1])
                acc_top_list.append(num)

        if len(acc_top_list):
            acc_list = metrics.accuracy(self.y, self.label, topk=tuple(acc_top_list))
            for num, acc in zip(acc_top_list, acc_list):
                setattr(self, 's_metric_accuracy_' + str(num), acc)

    def cal_g_metric(self):
        if 'precision' in self.g_metric_names:
            self.g_metric_precision = metrics.precision(self.t_metric_cmatrix, 1)
        if 'recall' in self.g_metric_names:
            self.g_metric_recall = metrics.recall(self.t_metric_cmatrix, 1)
        if 'fscore' in self.g_metric_names:
            self.g_metric_fscore = metrics.f_score(self.t_metric_cmatrix, 1)
        if 'auc' in self.g_metric_names:
            self.g_metric_auc = metrics.average_multiclass_ovo_score(self.buffer_glabels, self.buffer_gscores)
        if 'ap' in self.g_metric_names:
            self.g_metric_ap = metrics.ap_score(self.buffer_glabels, self.buffer_gscores)
        if 'sen' in self.g_metric_names or 'spe' in self.g_metric_names:
            self.g_metric_sen, self.g_metric_spe = metrics.roc(self.buffer_glabels, self.buffer_gscores)

    def cal_t_metric(self):
        if 'cmatrix' in self.t_metric_names:
            self.t_metric_cmatrix = metrics.comfusion_matrix(self.buffer_gpreds, self.buffer_glabels,self.opt.class_num)

    def backward(self):
        self.update_metrics('local')
        loss = 0
        if not len(self.loss_names):
            return

        for name in self.loss_names:
            loss += getattr(self,'loss_' + name) / self.opt.grad_iter_size
        if isinstance(loss, float):
            return

        loss.backward()

    def validation(self, dataset, visualizer, valid_iter):
        self.eval()
        o_l_state = self.opt.l_state
        self.opt.l_state = 'valid'
        iter_time_meter = metrics.TimeMeter()
        data_time_meter = metrics.TimeMeter()

        data_time_meter.start()
        iter_time_meter.start()

        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record(n = self.opt.batch_size)
            iter_time_meter.start()
            self.set_input(data)
            self.validate_parameters()
            self.update_metrics('local')

            iter_time_meter.record()

            if i % self.opt.v_print_freq == 0:  # print training losses and save logging information to the disk
                visualizer.print_current_info(-1, i, self, iter_time_meter.val, data_time_meter.val)
                self.save_stat_info(local=True)

            data_time_meter.start()
            iter_time_meter.start()

        if self.opt.multi_vsets > 1:
            combine_with_id(self)

        self.update_metrics('global')
        visualizer.print_global_info(-1, -1, self, iter_time_meter.sum/60, data_time_meter.sum/60)
        visualizer.plot_global_info(self, valid_iter, ptype='valid')
        self.train()
        self.save_stat_info()
        self.opt.l_state = o_l_state
        # self.reset_meters()
        self.clear_info()

    def plot_special_info(self):
        pass

    def print_special_info(self,log_name):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.eval()

    def train(self):
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def get_metric(self,metric_name_list):
        if 'accuracy_1' in metric_name_list and 'accuracy_3' in metric_name_list:
            return self.meters['accuracy_1'].avg, self.meters['accuracy_3'].avg
        elif 'accuracy_1' in metric_name_list:
            return self.meters['accuracy_1'].avg, 0
        value = 0
        count = 0
        for metric_name in metric_name_list:
            if metric_name in self.loss_names:
                value += self.meters[metric_name].avg
                #value = self.meters[metric_name].avg if value > self.meters[metric_name].avg else value
            elif metric_name in self.s_metric_names:
                value += self.meters[metric_name].avg
                #value = self.meters[metric_name].avg if value > self.meters[metric_name].avg else value
            elif metric_name in self.g_metric_names:
                value += float(getattr(self, 'g_metric_' + metric_name))
                #value = float(getattr(self, 'g_metric_' + metric_name)) if value > self.meters[metric_name].avg else value
        #assert value != 0, "The metric is error"
        #return value / len(metric_name_list)
        #return value
        return count, value / len(metric_name_list)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            # if self.opt.lr_policy == 'plateau' or self.opt.lr_policy == 'eco':
            if self.opt.lr_policy in metric_schedulers:
                tmp_metric = self.get_metric(self.scheduler_metric)
                m_kind = self.get_metric(self.scheduler_metric)
                if m_kind == 'loss':
                    tmp_metric = -tmp_metric
                scheduler.step(tmp_metric)
            else:
                scheduler.step()

    def change_dir_name(self, visualizer):
        count, value = self.get_metric(self.valid_metric)
        c_save_dir = self.save_dir

        if value <= 1:
            value_str = '{:.4f}'.format(value)
        else:
            value_str = '{:.3f}'.format(value)

        self.save_dir = self.o_save_dir #+ '_' + value_str
        self.opt.save_dir = self.save_dir
        os.system('mv ' + c_save_dir + ' ' + self.save_dir)
        visualizer.log_name = visualizer.log_name.replace(c_save_dir, self.save_dir)

        c_log_dir = c_save_dir.replace('checkpoints/', 'runs/')
        new_log_dir = self.save_dir.replace('checkpoints/', 'runs/')
        os.system('mv ' + c_log_dir + ' ' + new_log_dir)
        visualizer.writer.log_dir = new_log_dir

    def save_networks(self, epoch, visualizer):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        def save_nets():
            for name in self.net_names:
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    net = getattr(self, 'net_' + name)

                    if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                        torch.save(net.module.state_dict(), save_path)
                    else:
                        torch.save(net.state_dict(), save_path)

        if epoch != 'optimal':
            save_nets()
            return

        top1_acc, top3_acc = self.get_metric(self.valid_metric)
        if top3_acc > self.best_top3_acc or (top3_acc == self.best_top3_acc and top1_acc > self.best_top1_acc):
            self.best_top1_acc = top1_acc
            self.best_top3_acc = top3_acc
            self.change_dir_name(visualizer)

            for log_name in ['pred_result.txt', 'valid_buffer.json', 'valid_result.json', 'labels_scores.json']:
                pred_fname = osp.join(self.save_dir, log_name)
                if osp.exists(pred_fname):
                    n_pred_fname = pred_fname.replace(log_name, 'optimal_' + log_name)
                    os.system('mv ' + pred_fname + ' ' + n_pred_fname)

            self.wait_epoch = 0
            save_nets()
        #elif count == self.best_count and tmp_v_value > self.best_m_value:
        #    self.best_m_value = tmp_v_value
        #    self.change_dir_name(visualizer)
        #
        #    for log_name in ['pred_result.txt', 'valid_buffer.json', 'valid_result.json', 'labels_scores.json']:
        #        pred_fname = osp.join(self.save_dir, log_name)
        #        if osp.exists(pred_fname):
        #            n_pred_fname = pred_fname.replace(log_name, 'optimal_' + log_name)
        #            os.system('mv ' + pred_fname + ' ' + n_pred_fname)
        #
        #    self.wait_epoch = 0
        #    save_nets()

        else:
            self.wait_epoch += 1
            if self.wait_epoch > self.opt.patient_epoch:
                self.wait_over = True

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, load_dir ='', strict = True):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.net_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if load_dir == '':
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)

                if not osp.exists(load_path):
                    print('net ' + name + ' has no state dict')
                    continue

                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location = 'cpu')
                net.load_state_dict(state_dict, strict = strict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_stat_info(self, local = False):
        f = open(osp.join(self.save_dir, 'pred_result.txt'), 'w')
        for i in range(len(self.buffer_gscores)):
            label = self.buffer_glabels[i]
            pred = self.buffer_gscores[i]
            f.write(str(i) + ' ' + str(label) + ' ' + str(pred) + ' ' + str(self.buffer_ginput_ids[i]) + '\n')
        f.close()

        if not local:
            result_dict = {}
            ntype_list = ['loss','s_metric','g_metric','t_metric']
            for ntype in ntype_list:
                name_list = getattr(self,ntype + '_names')
                for name in name_list:
                    value = getattr(self, ntype + '_' + name)
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().detach().numpy()
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    result_dict[name] = value

            with open(osp.join(self.save_dir, self.opt.l_state + '_result.json'), 'w') as f:
                json.dump(result_dict, f)

        with open(osp.join(self.save_dir, self.opt.l_state + '_buffer.json'), 'w') as f:
            json.dump([self.buffer_ginput_ids, self.buffer_glabels, self.buffer_gscores], f)

    def next_epoch(self):
        pass

    def get_metric_kind(self, m_name):
        if m_name in self.loss_names:
            return 'loss'
        elif m_name in self.s_metric_names:
            return 's_metric'
        elif m_name in self.t_metric_names:
            return 't_metric'
        elif m_name in self.g_metric_names:
            return 'g_metric'

        AssertionError(False, 'This metric is not in this model')

    def detach_from_gpu(self):

        def detach_e(v):
            if isinstance(v, torch.Tensor) and 'cpu' not in v.device.type.lower():
                # return v.detach()
                return None
            elif isinstance(v, list):
                return [detach_e(sub_v) for sub_v in v]
            elif isinstance(v, tuple):
                return (detach_e(sub_v) for sub_v in v)
            else:
                return v

        k_list = list(filter(lambda x: not x.startswith('__'),dir(self)))
        for k in k_list:
            v = getattr(self, k)
            setattr(self,v, detach_e(v))
        torch.cuda.empty_cache()

    # def set_train_load_dir(self):
    #     pass
    #
    # def set_valid_load_dir(self):
    #     pass

    def clip_grad(self):
        for net_name in self.net_names:
            net = getattr(self, 'net_' + net_name)
            torch.nn.utils.clip_grad_norm(net.parameters(), 1, norm_type=2)
