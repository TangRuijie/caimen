import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import roi_pool
import torch
import copy
from . import resnet
from .or25d_transformer_net import init_weights, PositionalEncoder
from .or25d_transformer_net import MyTail, comb_feature

resnet_dict = {"ResNet18": resnet.resnet18, "ResNet34":resnet.resnet34, "ResNet50":resnet.resnet50, "ResNet101":resnet.resnet101, "ResNet152":resnet.resnet152}
feature_num_dict = {"ResNet18": 512, "ResNet34": 512, "ResNet50": 2048, "ResNet101": 2048, "ResNet152": 2048}


class ResNet(nn.Module):
    def __init__(self, net_name = 'ResNet50', bottleneck_dim = 256, class_num = 2, with_fc = True, **kwargs):
        super(ResNet, self).__init__()  # call the initialization method of BaseModel
        self.num_features = kwargs['fpn_channels']
        self.channel_comb_type = ''
        if 'channel_comb_type' in kwargs.keys():
            self.channel_comb_type = kwargs['channel_comb_type']

        resnet = resnet_dict[net_name](pretrained=True)

        if 'in_channel' in kwargs.keys():
            tmp_conv = self.conv1 = nn.Conv2d(kwargs['in_channel'], 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.RCNN_layer0 = nn.Sequential(tmp_conv, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            self.RCNN_layer0 = nn.Sequential(resnet.conv, resnet.bn1, resnet.relu, resnet.maxpool)

        self.RCNN_layer1 = nn.Sequential(resnet.layer1)
        self.RCNN_layer2 = nn.Sequential(resnet.layer2)
        self.RCNN_layer3 = nn.Sequential(resnet.layer3)
        self.RCNN_layer4 = nn.Sequential(resnet.layer4)

        self.RCNN_toplayer = nn.Conv2d(2048, self.num_features, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(1024, self.num_features, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d( 512, self.num_features, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d( 256, self.num_features, kernel_size=1, stride=1, padding=0)

        self.maxpool2d = nn.MaxPool2d(1, stride=2)

        batch_size = kwargs['batch_size']
        t = kwargs['seq_len']

        self.embd_pos = kwargs['embd_pos']
        self.head_block_list = kwargs['head_block_list']
        self.up_layer_size = kwargs['up_layer_size']
        self.context_frames = kwargs['context_frames']
        self.input_size = kwargs['load_size']
        self.sub_d_model = kwargs['sub_d_model']
        self.crop_ratio = kwargs['crop_ratio']
        self.block_comb_mode = kwargs['block_comb_mode']
        self.multi_task = kwargs['multi_task']
        self.pos_embd = PositionalEncoder(self.num_features, max_seq_len=t, batch_size=batch_size, t=t)
        self.bottleneck_dim = bottleneck_dim
        self.with_fc = with_fc

        self.resize_ratio_list = []
        self.roi_out_size_list = []

        for i in range(4):
            if not self.head_block_list[i]:
                self.resize_ratio_list.append(-1)
                self.roi_out_size_list.append(-1)
                continue

            if self.channel_comb_type == 'conv':
                tmp_conv = nn.Conv2d(in_channels=512, out_channels= 256, kernel_size=1)
                self.add_module('channel_comb_conv' + str(i), tmp_conv)

            spatial_size = self.up_layer_size * (2**(3 - i))
            resize_ratio = spatial_size/self.input_size
            roi_out_size = int(spatial_size * self.crop_ratio)
            self.resize_ratio_list.append(resize_ratio)
            self.roi_out_size_list.append(roi_out_size)

            tmp_num_feature = self.num_features
            if self.channel_comb_type == 'cat':
                tmp_num_feature *= 2
            tail = MyTail(d_model = self.sub_d_model, crop_size = roi_out_size, roi_out_size = roi_out_size,
                        num_features = tmp_num_feature, **kwargs)

            self.add_module('tail' + str(i), tail)

        self.roi_op = kwargs['roi_op']

        in_features = self.sub_d_model
        if kwargs['block_comb_mode'] == 'cat':
            in_features *= sum(self.head_block_list)
        if kwargs['frame_comb_mode'] == 'cat':
            in_features *= kwargs['context_frames']
        if kwargs['head_comb_mode'] == 'cat':
            in_features *= kwargs['head_number']

        if 'add_info' in kwargs.keys():
            self.add_info = kwargs['add_info']
        else:
            self.add_info = '000' #loc, age, gender

        if self.add_info[0] != '0':
            in_features += 4
        if self.add_info[1] != '0':
            in_features += 1
        if self.add_info[2] != '0':
            in_features += 3
        self.in_features = in_features

        if with_fc:
            if self.bottleneck_dim > 0:
                self.bottleneck = nn.Linear(in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = in_features

        if self.multi_task:
            self.bottleneck2 = nn.Linear(in_features, bottleneck_dim)
            self.fc2 = nn.Linear(bottleneck_dim, 3)
            self.bottleneck2.apply(init_weights)
            self.fc2.apply(init_weights)

    def trans_bboxes(self, bboxes_data, resize_ratio):
        new_bboxes_data = []
        context_range_list = []

        for i in range(bboxes_data.shape[0]):
            for j in range(bboxes_data.shape[1]):
                tmp_bboxes = bboxes_data[i, j]
                for bbox in tmp_bboxes:
                    if bbox[0] >= 0:
                        tmp_ind = i * bboxes_data.shape[1] + j
                        tmp_min = max(0, j - self.context_frames // 2) + i * bboxes_data.shape[1]
                        tmp_max = min(bboxes_data.shape[1] - 1, j + self.context_frames // 2) + i * bboxes_data.shape[1]
                        context_range_list.append([i, tmp_min, tmp_max])
                        new_bboxes_data.append([tmp_ind] + list(bbox * resize_ratio))

        # for i, tmp_bboxes in enumerate(bboxes_data):
        #     for bbox in tmp_bboxes:
        #         if bbox[0] >= 0:
        #             new_bboxes_data.append([i] + list(bbox * resize_ratio))
        return new_bboxes_data, context_range_list

    def expand_x_with_K(self,x, k_list):
        tmp_list = []
        for k in k_list:
            k = int(k)
            tmp_list.append(x[k])
        new_x = torch.stack(tmp_list)
        return new_x

    def compress_x_with_K(self,x, k_list, ins_num):

        inds_data = [[] for _ in range(ins_num)]
        for i, k in enumerate(k_list):
            inds_data[int(k)].append(i)

        new_x = []
        for i, inds in enumerate(inds_data):
            if len(inds) > 0:
                tmp_data = x[inds].mean(dim = 0)
            else:
                tmp_data = torch.zeros((self.sub_d_model,)).cuda(x.device)

            new_x.append(tmp_data)
        new_x = torch.stack(new_x, dim = 0)
        return new_x

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x, or_bboxes, mask):
        """
        :param x: B * T * H * W * C
        :param bboxes: tensor B * T * N * [x1 , y1 , x2 , y2]
        :return:
        """
        tmp_shape = x.shape
        if len(tmp_shape) == 5:
            n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
            mask_shape = [tmp_shape[0] * tmp_shape[1], 1, tmp_shape[3], tmp_shape[4]]
            x = x.view(n_shape)
            mask = mask.view(mask_shape)


        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(x)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)  # Bx256x8x8
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)   # Bx256x16x16
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)   # Bx256x32x32
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)   # Bx256x64x64
        p6 = self.maxpool2d(p5)

        '''
        step 1: construct bboxes        
        '''

        roi_x_list = []
        for i, p in enumerate([p2,p3,p4,p5]):
            if not self.head_block_list[i]:
                continue
            bboxes = copy.copy(or_bboxes)
            bboxes, context_range_list = self.trans_bboxes(bboxes, self.resize_ratio_list[i])
            bboxes = torch.Tensor(bboxes)
            bboxes = bboxes.cuda(device=x.device)

            if self.channel_comb_type != 'no':
                p_shape = tuple(p.shape[-2:])
                tmp_mask = F.interpolate(mask, p_shape)
                mask_p = p * tmp_mask

                if self.channel_comb_type == 'conv':
                    tmp_p = torch.cat([p, mask_p], dim = 1)
                    tmp_conv = getattr(self, 'channel_comb_conv' + str(i))
                    p = tmp_conv(tmp_p)
                elif self.channel_comb_type == 'add':
                    p = (p + mask_p) / 2
                else:
                    p = torch.cat([p, mask_p], dim = 1)

            '''
            step 2: roi bboxes
            '''
            if bboxes.shape[0] == 0:
                roi_x = torch.zeros(tmp_shape[0], self.in_features).cuda(x.device)
            else:
                if self.embd_pos:
                    p = self.pos_embd(p)
                roi_out_size = self.roi_out_size_list[i]
                roi_x = roi_pool(p, bboxes, [roi_out_size, roi_out_size], 1)
                tail = getattr(self, 'tail' + str(i))
                roi_x = tail(p, roi_x, context_range_list)

            roi_x_list.append(roi_x)

        x_list = []
        x = comb_feature(roi_x_list, mode=self.block_comb_mode)
        x_list.append(x)

        info_data = []
        for i in range(x.shape[0]):
            info_x_list = []
            info_data.append(info_x_list)

        info_x = torch.Tensor(info_data).cuda(x.device)
        x = torch.cat([x, info_x], dim = -1)

        if self.with_fc:
            x1 = self.bottleneck(x)
            x_list.append(x1)
            y = self.fc(x1)
            data_list = [y, x_list]

            if self.multi_task:
                x2 = self.bottleneck2(x)
                y2 = self.fc2(x2)
                data_list.append(y2)

            return data_list

        else:
            return x
