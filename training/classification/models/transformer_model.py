import torch
import copy
from util import *
import cv2
import os 
import numpy as np 
# from imgaug import augmenters as iaa
import json
import torch.nn.functional as F
import torch.nn as nn
from .base_model import BaseModel
from .networks.or25d_transformer_net import ResNet as TransNet
from .networks.fpn_transformer_net import ResNet as FPNNet
from .networks.multimilheat_net import MultiMILNet as HeatNet
from util.loss import BalancedLoss
from util.loss import FocalLoss
import torch.distributed as distrib

class TransformerModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):

        parser.add_argument('--dif_lr', type=int, default=0)
        parser.add_argument('--max_bbox_num', type=int, default=1)
        parser.add_argument('--add_bbox_noise', type=int, default=1)
        parser.add_argument('--k_mode', type=str, default='conv')
        parser.add_argument('--add_conv', type=int, default=0)
        parser.add_argument('--ex_method_name', type=str, default='')
        parser.add_argument('--with_tail_fc', type=int, default=0)
        parser.add_argument('--multi_crop_size', type=int, default=1)
        parser.add_argument('--vis_trunc_ratio', type=float, default=0)
        parser.add_argument('--head_block_list', type=str, default='[0,1,1,1]')
        # parser.add_argument('--roi_layer_list', type=str, default='[1,1,1,1]')
        parser.add_argument('--crop_ratio', type=float, default=0.3)
        parser.add_argument('--crop_stride_list', type=str, default='[1,1,1,1]')
        parser.add_argument('--sub_d_model', type=int, default=128)
        parser.add_argument('--learning_inter', type=float, default=0.1)

        parser.add_argument('--in_channel', type=int, default=3)
        parser.add_argument('--heat_ratio', type=float, default=0.5)
        parser.add_argument('--up_layer_size', type=int, default=14)
        parser.add_argument('--duel_item', type=int, default=1)
        parser.add_argument('--block_dropout', type=float, default=0)
        parser.add_argument('--model_input_size', type=int, default=20)
        parser.add_argument('--use_weighted_scores', type=int, default=1)
        parser.add_argument('--inference_input_size', type=int, default=16)
        parser.add_argument('--reinit_data', type=int, default=10)
        parser.add_argument('--always_eval', type=int, default=0)
        parser.add_argument('--fc_mult', type=int, default=10)
        parser.add_argument('--tmp_train_num', type=int, default=0)
        parser.add_argument('--data_degree', type=str, default='hard')
        parser.add_argument('--net_name', type=str, default="ResNet50", help='resnet name')
        parser.add_argument('--use_bottleneck', type=bool, default=True, help='use_bottleneck')
        parser.add_argument('--positive_score_thred', type=float, default=0)
        parser.add_argument('--bottleneck_dim', type=int, default=256, help='bottleneck dim')
        parser.add_argument('--out_channel', type=int, default=2)
        parser.add_argument('--with_sig', type=int, default=1)
        parser.add_argument('--use_heat', type=int, default=0)
        parser.add_argument('--loss_type', type=str, default='focal')
        parser.add_argument('--focal_alpha', type=float, default=0.6)
        parser.add_argument('--focal_gamma', type=float, default=2)
        parser.add_argument('--use_softmax', type=int, default=1)
        parser.add_argument('--use_sigmoid', type=int, default=0)
        parser.add_argument('--weight_len', type=int, default=0)
        parser.add_argument('--inner_bs', type=int, default=20)
        parser.add_argument('--use_smooth', type=int, default=1)
        parser.add_argument('--use_heat', type=int, default=1)
        parser.add_argument('--smooth_ratio', type=float, default=0.01)
        parser.add_argument('--neg_lr_ratio', type=float, default=0.001)
        parser.add_argument('--use_heatmap',type=int,default = 0)
        parser.add_argument('--cbp_kernel_sizes',type=str,default = '[3,1]')
        parser.add_argument('--method_name',type=str,default = 'trans')
        parser.add_argument('--heat_layer',type=int,default = 4)
        parser.add_argument('--tail_head_num',type=int,default = 1)
        parser.add_argument('--block_layer',type=int,default = 2)
        parser.add_argument('--roi_op',type=str,default = 'directly')
        parser.add_argument('--roi_out_size',type=int,default = 2)
        parser.add_argument('--detach_roi',type=int,default = 0)
        parser.add_argument('--embd_pos',type=int,default = 0)
        parser.add_argument('--context_frames',type=int,default = 3)
        parser.add_argument('--heat_comb_mode',type=str,default = 'cat')
        parser.add_argument('--channel_comb_type',type=str,default = 'no')
        parser.add_argument('--consider_distance',type=int,default = 0)

        parser.add_argument('--block_comb_mode',type=str,default = 'cat')
        parser.add_argument('--frame_comb_mode',type=str,default = 'mean')
        parser.add_argument('--head_comb_mode',type=str,default = 'cat')
        parser.add_argument('--fpn_channels',type=int,default = 256)
        parser.add_argument('--add_info',type=str,default ='000', help = 'loc age gender')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.vis_count = 0
        self.file_created = False

        opt.head_block_list = str2json(opt.head_block_list)
        res_method_dict = {'trans': TransNet, 'fpn': FPNNet, 'heat': HeatNet}

        up_layer_size_dict = {256:8, 384: 12, 448:14, 512:16}

        params_dict = {'cbp': {'opt':opt},
                       'heat': {'use_or_heat':opt.use_heat,
                                'load_size': opt.load_size,
                                'up_layer_size': up_layer_size_dict[opt.load_size],
                                'heat_ratio': opt.heat_ratio,
                                'heat_comb_mode': opt.heat_comb_mode,
                                },
                       'trans': {
                            'batch_size': opt.batch_size,
                            'load_size': opt.load_size,
                            'seq_len': opt.resize_dcm_len,
                            'embd_pos': opt.embd_pos,
                            'heat_layer': opt.heat_layer,
                            'up_layer_size': up_layer_size_dict[opt.load_size],
                            'crop_ratio': opt.crop_ratio,
                            'context_frames': opt.context_frames,
                            'roi_op': opt.roi_op,
                            'frame_comb_mode': 'mean',
                            'head_comb_mode': 'cat',
                            'sub_d_model': opt.sub_d_model,
                            'block_layer': opt.block_layer,
                            'head_number': opt.tail_head_num},
                       'fpn': {
                           'in_channel': opt.in_channel,
                           'batch_size': opt.batch_size,
                           'load_size': opt.load_size,
                           'seq_len': opt.resize_dcm_len,
                           'embd_pos': opt.embd_pos,
                           'heat_layer': opt.heat_layer,
                           'up_layer_size': up_layer_size_dict[opt.load_size],
                           'crop_ratio': opt.crop_ratio,
                           'context_frames': opt.context_frames,
                           'roi_op': opt.roi_op,
                           'add_info': opt.add_info,
                           'fpn_channels': opt.fpn_channels,
                           'head_block_list': opt.head_block_list,
                           'frame_comb_mode': 'mean',
                           'channel_comb_type': opt.channel_comb_type,
                           'block_comb_mode': 'cat',
                           'head_comb_mode': 'cat',
                           'sub_d_model': opt.sub_d_model,
                           'block_layer': opt.block_layer,
                           'multi_task': opt.multi_task,
                           'head_number': opt.tail_head_num}
                       }
        self.net_res = res_method_dict[opt.method_name](net_name = opt.net_name, bottleneck_dim = opt.bottleneck_dim,
                                                       class_num = self.opt.class_num, **params_dict[opt.method_name])
        #self.net_res = nn.SyncBatchNorm.convert_sync_batchnorm(self.net_res)
        self.opt = opt

        self.net_names = ['res']

        if self.opt.class_num == 2:
            self.s_metric_names = ['accuracy_1']
            self.g_metric_names = ['auc_' + str(i) for i in range(self.opt.class_num)]
            self.valid_metric = ['accuracy_1']
            self.scheduler_metric = 'accuracy_1'
        else:
            self.s_metric_names = ['accuracy_1', 'accuracy_3']
            self.g_metric_names = ['auc_' + str(i) for i in range(self.opt.class_num)]
            self.valid_metric = ['accuracy_1', 'accuracy_3']
            self.scheduler_metric = 'accuracy_1'
        '''loss setting'''

        if 'focal' in self.opt.loss_type:
            min_num = min(self.opt.sample_num_list)
            alpha_list = 1 / np.array(self.opt.sample_num_list) * min_num
            alpha_list = alpha_list.tolist()

            self.criterion_focal = FocalLoss(gamma = self.opt.focal_gamma, alpha = alpha_list,
                                             device=self.opt.gpu_ids[0], size_average=True)

        if 'cross' in self.opt.loss_type:
            self.criterion_cross = nn.CrossEntropyLoss()

        if 'balanced' in self.opt.loss_type and self.opt.l_state == 'train':
            print('class num',len(self.opt.sample_num_list))
            sample_num_list = torch.Tensor(self.opt.sample_num_list).cuda()
            distrib.all_reduce(sample_num_list, op=distrib.ReduceOp.SUM)
            sample_num_list = sample_num_list.tolist() # convert from tensor to list
            self.criterion_balanced = BalancedLoss(no_of_classes=len(self.opt.sample_num_list), samples_per_cls=sample_num_list, loss_type=opt.balanced_loss_type)
            self.criterion_balanced = self.criterion_balanced.cuda(self.opt.gpu_ids[0])

        if self.opt.multi_task and self.opt.l_state == 'train':
            self.loss_names.append('c2')
            self.opt.class_num = len(self.opt.sample_num_list)
            self.criterion_balanced2 = BalancedLoss(no_of_classes=len(self.opt.sample_num_list2), samples_per_cls=self.opt.sample_num_list2, loss_type=opt.balanced_loss_type)
            self.criterion_balanced2 = self.criterion_balanced2.cuda(self.opt.gpu_ids[0])

        self.buffer_ys = []

    @staticmethod
    def supply_option_info(opt):

        print('step 0')
        if opt.l_state == 'train':
            TransformerModel.set_train_load_dir(opt)
        else:
            TransformerModel.set_valid_load_dir(opt)
        return opt

    def set_input(self, data):
        self.input, self.label, self.label2, self.input_id, self.heatmap, self.loc, self.age, self.gender, self.mask, self.bbox = data
        self.input_id = list(self.input_id)
        self.input = self.input.cuda(self.gpu_ids[0]).float()
        self.mask = self.mask.cuda(self.gpu_ids[0]).float()
        self.label = self.label.cuda(self.gpu_ids[0])
        self.label2 = self.label2.cuda(self.gpu_ids[0])
        self.heatmap = self.heatmap.cuda(self.gpu_ids[0]).float()
        self.input_size = self.input.shape[0]

        # print('self. bbox', self.bbox)

    def average_weights(self, dataset_size):
        e = dataset_size
        e_ = torch.tensor([dataset_size]).cuda()
        distrib.all_reduce(e_, op=distrib.ReduceOp.SUM)

        for name, param in self.net_res.named_parameters():
            if param.requires_grad:
                param.data *= e
                distrib.all_reduce(param.data, op=distrib.ReduceOp.SUM)
                param.data /= e_.float()

    def check_early_stop(self, es_signal):
        es_num = torch.tensor([es_signal]).cuda()
        distrib.all_reduce(es_num, op=distrib.ReduceOp.SUM)
        world_size = torch.tensor([1]).cuda()
        if es_num >= world_size:
            return True
        else:
            return False

    def check_finish(self, finish_signal):
        f_num = torch.tensor([finish_signal]).cuda()
        distrib.all_reduce(f_num, op=distrib.ReduceOp.SUM)
        world_size = distrib.get_world_size()
        world_size = torch.tensor([world_size]).cuda()
        if f_num >= world_size:
            return True
        else:
            return False


    def video_forward(self):
        tmp_input = self.input
        tmp_heatmap = self.heatmap
        tmp_bboxes = self.trans_bboxes(self.bbox)

        # print('tmp bboxes shape',tmp_bboxes.shape)
        input_dict = {'heat': [tmp_input, tmp_bboxes],
                      'cbp': [tmp_input, tmp_heatmap, tmp_bboxes],
                      'trans': [tmp_input, tmp_bboxes],
                      'fpn': [tmp_input, tmp_bboxes],
                      }

        if self.opt.multi_task:
            tmp_y, self.feature_list, tmp_y2 = self.net_res(*input_dict[self.opt.method_name], loc = self.loc, age = self.age, gender = self.gender, mask = self.mask)
        else:
            tmp_y, self.feature_list = self.net_res(*input_dict[self.opt.method_name], loc = self.loc, age = self.age, gender = self.gender, mask = self.mask)

        if self.opt.use_softmax:
            tmp_y = F.softmax(tmp_y, dim = 1)

            if self.opt.multi_task:
                tmp_y2 = F.softmax(tmp_y2, dim = 1)
                self.y2 = tmp_y2
        self.y = tmp_y

        if self.opt.l_state != 'train' and self.opt.vis_method.lower() == 'gradcam':
            self.vis_method.cal_grad(tmp_y, self.label)

        self.score = self.y[:, 1] # self.score is useless
        self.score = self.score.view(-1,)

    def forward(self):
        self.video_forward()

    def cal_loss(self):
        if self.opt.l_state != 'train':
            self.loss_c = -1
            return

        if self.opt.loss_type == 'focal':
            loss_c = self.criterion_focal(self.y, self.label)

        elif self.opt.loss_type == 'balanced':
            loss_c = self.criterion_balanced(self.y, self.label, reduce = False)

        elif self.opt.loss_type == 'cross':
            loss_c = self.criterion_cross(self.y, self.label)

        if self.opt.consider_distance:
            pred = torch.argmax(self.y, 1)
            w_index = list(zip(self.label, pred))
            weights = self.dis_matrix[w_index]
            loss_c *= weights.unsqueeze(-1).float()

        self.loss_c = torch.mean(loss_c)

        if self.opt.multi_task:
            self.loss_c2 = self.criterion_balanced2(self.y2, self.label2) * self.opt.multi_task_ratio

    def get_parameters(self):
        if self.opt.method_name == 'trans':
            parameter_list = [{"params": self.net_res.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 1}, \
                              {"params": self.net_res.fc.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult}]
            parameter_list.append({"params": self.net_res.tail.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
            if self.net_res.bottleneck_dim > 0:
                parameter_list.append({"params": self.net_res.bottleneck.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
        else:
            if not self.opt.dif_lr:
                parameter_list = [{"params": self.net_res.parameters(), "lr_mult": 1, 'decay_mult': 1}]
            else:
                parameter_list = []
                for i in range(5):
                    tmp_module = getattr(self.net_res, 'RCNN_layer' + str(i))
                    parameter_list.append({"params": tmp_module.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})

                parameter_list.append({"params":self.net_res.RCNN_toplayer.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})

                for i in range(1, 4):
                    tmp_module = getattr(self.net_res, 'RCNN_smooth' + str(i))
                    parameter_list.append({"params": tmp_module.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
                    tmp_module = getattr(self.net_res, 'RCNN_latlayer' + str(i))
                    parameter_list.append({"params": tmp_module.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})

                parameter_list.append({"params": self.net_res.fc.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
                if self.net_res.bottleneck_dim > 0:
                    parameter_list.append({"params": self.net_res.bottleneck.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})

                if self.opt.multi_task:
                    parameter_list.append({"params": self.net_res.fc2.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})
                    if self.net_res.bottleneck_dim > 0:
                        parameter_list.append({"params": self.net_res.bottleneck2.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})

                for i in range(4):
                    if self.opt.head_block_list[i]:
                        tail = getattr(self.net_res, 'tail' + str(i))
                        parameter_list.append({"params": tail.parameters(), "lr_mult": self.opt.fc_mult, 'decay_mult': self.opt.fc_mult})


        return parameter_list

    def stat_info(self):
        super(TransformerModel, self).stat_info()
        tmp_list = self.y.detach().cpu().numpy().tolist()
        self.buffer_ys.extend(tmp_list)

    def cal_g_metric(self):
        if 'auc' in self.g_metric_names:
            super(TransformerModel, self).cal_g_metric()
        else:
            num_list = []
            for m_name in self.g_metric_names:
                if m_name.startswith('auc'):
                    num = int(m_name.split('_')[-1])
                    num_list.append(num)
            
            if len(num_list):
                ys_matrix = np.array(self.buffer_ys)
                assert ys_matrix.shape[0] == len(self.buffer_glabels)
                ys_labels = np.array(self.buffer_glabels)
                for num in num_list:
                    scores = ys_matrix[:,num]
                    labels = ys_labels == num
                    labels = labels.astype('long')
                    # print('label shape', labels.shape)
                    # print('scores shape', scores.shape)
                    auc = metrics.auc_score(labels, scores)
                    setattr(self, 'g_metric_auc_' + str(num), auc)
                pass


    def visualize(self):
        if self.opt.l_state == 'train':
            return

        if self.opt.vis_method == 'orimg':
            tmp_names = []
            if self.pred.item() != self.label.item():
                tmp_names.append(str(self.label.item()) + '_' +str(self.pred.item()) +  '_' + self.input_id[0])
                self.vis_method.show_or_image(self.input, intype='NCHW', name_list= tmp_names, to_one_img=True, bbox_data=self.bbox)

        elif self.opt.vis_method == 'show_weights':
            self.weights_show()

        elif self.opt.vis_method == 'show_weights_v2':
            self.weights_show_v2()

    def trans_bboxes(self,old_bboxes_data):
        bboxes_data = copy.deepcopy(old_bboxes_data)
        max_len = self.opt.max_bbox_num
        new_bboxes = []

        for seq_bbox_data in bboxes_data:
            new_seq_bbox_data = []
            for tmp_bboxes in seq_bbox_data:
                if len(tmp_bboxes) > max_len:
                    tmp_bboxes = tmp_bboxes[:max_len]
                if len(tmp_bboxes) < max_len:
                    while len(tmp_bboxes) < max_len:
                        tmp_bboxes.append([-1,-1,-1,-1])
                new_seq_bbox_data.append(tmp_bboxes)
            new_bboxes.append(new_seq_bbox_data)
        # import ipdb
        # ipdb.set_trace()
        try:
            bboxes_data = torch.Tensor(new_bboxes)
        except:
            print('input id')
            print(self.input_id)
            print('new bbox ')
            print(new_bboxes)
        return bboxes_data

    def weights_show_v2(self):
        def cal_cam(roi_weight, layer_size, crop_size, stride):
            if roi_weight is None:
                return None

            if isinstance(roi_weight, torch.Tensor):
                roi_weight = roi_weight.cpu().numpy()

            # print('roi weight shape ', roi_weight.shape)
            tmp_size = roi_weight.shape[1]

            if layer_size == 56:
                crop_num = (layer_size - crop_size + 1) + 1

            else:
                crop_num = (layer_size - crop_size)//stride + 1
            # assert crop_num**2 == tmp_size

            # print(layer_size)
            # print(stride)
            cam = np.zeros((roi_weight.shape[0], layer_size, layer_size))
            count_cam = np.zeros((roi_weight.shape[0], layer_size, layer_size))
            # print('roi weight shape ', roi_weight.shape)
            # print('cam shape ', cam.shape)
            # print('crop num ', crop_num)


            # tmp_ratio = np.max(roi_weight, axis=1,keepdims=True)
            # roi_weight /= tmp_ratio
            if self.opt.method_name == 'pytrans':
                roi_weight = roi_weight.reshape((roi_weight.shape[0], crop_num, crop_num))
            else:
                roi_weight = roi_weight.reshape((roi_weight.shape[0], layer_size, layer_size))

            # for i in range(crop_num):
            #     for j in range(crop_num):
            #         cam[:, i * stride: i * stride + crop_size, j * stride: j * stride + crop_size] += roi_weight[:,i,j][:,np.newaxis][:,:,np.newaxis]
            #         count_cam[:, i * stride: i * stride + crop_size, j * stride: j * stride + crop_size] += 1
            #
            # cam /= count_cam

            cam = roi_weight

            # print('cam shape ', cam.shape)
            tmp_ratio = np.max(cam, axis=1,keepdims=True)
            tmp_ratio = np.max(tmp_ratio, axis=2,keepdims=True)
            cam = cam/tmp_ratio


            cam = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
                # print('tmp_ratio ', tmp_ratio)
                # print('cam shape ', cam.shape)
                # print('crop size ', crop_size)
                # print('max cam  ', np.max(cam))
                # print('min cam  ', np.min(cam))

            return cam

        bbox_list = self.bbox[self.max_ind]

        def bbox_num(n):
            num = 0
            for bboxes in self.bbox[:n]:
                num += min(len(bboxes), self.opt.max_bbox_num)
            return num

        if len(bbox_list) > self.opt.max_bbox_num:
            bbox_list = bbox_list[:self.opt.max_bbox_num]

        roi_weights_range = range(bbox_num(self.max_ind), bbox_num(self.max_ind+1))
        label = self.label.item()


        if label == 0:
            return

        if len(bbox_list) == 0:
            return

        if label == 1 and self.score.item() < 0.6:
            return


        crop_ratio_list = self.opt.crop_ratio_list
        crop_stride_list = self.opt.crop_stride_list
        crop_size_list = []
        layer_size_list = []
        up_layer_size = 14

        for i in range(4):
            layer_size = up_layer_size * (2**(3-i))
            crop_size = int(layer_size * crop_ratio_list[i])
            crop_size_list.append(crop_size)
            layer_size_list.append(layer_size)

        img_data = []

        images = tensor2im(self.input[self.max_ind: self.max_ind+1], norm_to_255=True)
        images = np.repeat(images, len(bbox_list), axis=0)

        or_image = copy.deepcopy(images[0])

        roi_layer_list = self.opt.roi_layer_list
        for roi_ind, tmp_roi_weight in enumerate(self.roi_weight):
            if not roi_layer_list[roi_ind] or tmp_roi_weight is None:
                continue
            # if roi_ind <= 1:
            #     continue

            # print('roi weight range ', roi_weights_range)
            # print('tmp roi weight shape ', tmp_roi_weight.shape)

            tmp_roi_weight = tmp_roi_weight[list(roi_weights_range)]

            # print('tmp roi weight shape after ', tmp_roi_weight.shape)
            cams_list = []

            for i in range(self.opt.block_layer):
                # tmp_roi_weight = tmp_roi_weight[..., i]
                cams = cal_cam(roi_weight=tmp_roi_weight[..., i], crop_size = crop_size_list[roi_ind],
                               layer_size = layer_size_list[roi_ind], stride = crop_stride_list[roi_ind])
                cams_list.append(cams)

            # if roi_ind == 1:
            #     iaa_resize = iaa.Resize({"height": 7, "width": 7}, interpolation="linear")
            #     tmp_cam = iaa_resize.augment_image(cams_list[0][0])
            #     print(tmp_cam)
            #     import ipdb
            #     ipdb.set_trace()

            cam_images = GradCam.show_cam_on_image(None, images, cams_list)
            tmp_img_list = []

            # import ipdb
            # ipdb.set_trace()

            for i, bbox in enumerate(bbox_list):
                bbox = [int(e.item()) for e in bbox]
                # print(bbox)
                tmp_img = cv2.rectangle(images[i], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cat_list = [tmp_img]
                for j in range(self.opt.block_layer):
                    cat_list.append(cam_images[j][i])
                new_imgs = np.concatenate(cat_list, axis=1)
                tmp_img_list.append(new_imgs)
            img_data.append(tmp_img_list)

        new_img_list = []
        for i in range(len(img_data[0])):
            tmp_list = []
            for j in range(len(img_data)):
                tmp_list.append(img_data[j][i])
            tmp_img = np.concatenate(tmp_list, axis=0)
            new_img_list.append(tmp_img)

        new_img = np.concatenate(new_img_list, axis=0)
        save_name = osp.join(self.vis_dir, str(self.vis_count) + '.jpg')
        or_name = osp.join(self.vis_dir, 'or_' + str(self.vis_count) + '.jpg')
        print('vis count ', self.vis_count)
        cv2.imwrite(save_name, new_img)
        # cv2.imwrite(or_name, or_image)

        self.vis_count += 1

    def next_epoch(self):
        pass
        # self.opt.dataset.dataset.shuffle_data()
        # pass
    @staticmethod
    def set_valid_load_dir(opt):
        if opt.load_dir != '':
            return

        load_dir_dict = {
            0: f'checkpoints/{opt.path}'
        }
        opt.load_dir = load_dir_dict[opt.load_dir_ind]

    @staticmethod
    def set_train_load_dir(opt):
        if opt.load_dir != '':
            return

        load_dir_dict = {
            0: f'checkpoints/{opt.path}'
        }
        opt.load_dir = load_dir_dict[opt.load_dir_ind]