import torch
import copy
from . import mynets
import torch.nn as nn
import torch.nn.functional as F
from .utils.weight_init import init_weights
from .layers.identity_layers import IdentityLayer

class MultiMILNet(nn.Module):
    def __init__(self, net_name = 'resnet50', class_num=2, with_fc = True,
                 pool_mode='gated_attention', top_k = -1, **kwargs):
        super(MultiMILNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = mynets.__dict__[net_name]()
        # self.backbone.avgpool = IdentityLayer()
        # self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.backbone.last_linear.in_features

        self.pool_mode = pool_mode
        self.top_k = top_k
        self.with_fc = with_fc
        self.context_frames = 2
        self.up_layer_size = kwargs['up_layer_size']
        self.input_size = kwargs['load_size']
        self.heat_comb_mode = kwargs['heat_comb_mode']
        self.heat_ratio = kwargs['heat_ratio']

        self.resize_ratio_list = []
        for i in range(4):
            spatial_size = self.up_layer_size * (2**(3 - i))
            resize_ratio = spatial_size/self.input_size
            self.resize_ratio_list.append(resize_ratio)

        if 'avgpool_mode' in kwargs.keys() and kwargs['avgpool_mode'] == 'max':
            self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.backbone.last_linear = IdentityLayer()

        if self.heat_comb_mode == 'cat':
            in_features *= 2

        self.bottleneck_dim = 256

        if self.pool_mode in ['attention', 'gated_attention']:
            self.attention_1 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_gate = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_2 = nn.Linear(self.bottleneck_dim, 1)
            
            self.attention_1.apply(init_weights)
            self.attention_gate.apply(init_weights)
            self.attention_2.apply(init_weights)

        self.bottleneck = nn.Linear(in_features, self.bottleneck_dim)
        self.fc = nn.Linear(self.bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)


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

    def attention(self, x):
        x1 = self.attention_1(x)
        x_out = F.tanh(x1)
        x2 = self.attention_2(x_out)
        return x2

    def gated_attention(self, x):
        x1 = self.attention_1(x)
        x_gate = F.sigmoid(self.attention_gate(x))
        x_out = F.tanh(x1) * x_gate
        x2 = self.attention_2(x_out)
        return x2

    def forward(self, input,  or_bboxes):

        tmp_shape = input.shape
        if len(tmp_shape) == 5:
            n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
            input = input.view(n_shape)

        x = self.backbone.features(input)
        # print('x shape ', x.shape)

        heat = torch.zeros(x.shape).cuda(x.device)
        bboxes = copy.copy(or_bboxes)
        bboxes, context_range_list = self.trans_bboxes(bboxes, self.resize_ratio_list[-1])

        for bbox in bboxes:
            ind, x1, y1, x2, y2 = bbox
            x1 = int(x1.item())
            y1 = int(y1.item())
            x2 = int(x2.item())
            y2 = int(y2.item())
            heat[ind,: y1:y2 + 1,:x1: x2 + 1] = 1

        heat = x * heat * self.heat_ratio
        if self.heat_comb_mode == 'cat':
            x = torch.cat([x, heat], dim = 1)
        else:
            x = x * heat

        # bboxes = torch.Tensor(bboxes)
        # bboxes = bboxes.cuda(device=x.device)
        x = self.avgpool(x).view(x.shape[0], -1)

        bt_x1 = self.bottleneck(x)
        y = self.fc(bt_x1)

        if len(tmp_shape) != 5:
            return y

        x = x.view(tmp_shape[0], tmp_shape[1], -1)
        bt_x1 = bt_x1.view(tmp_shape[0], tmp_shape[1], -1)
        y = y.view(tmp_shape[0], tmp_shape[1], -1)
        slice_y = y

        if self.top_k > 0:
            weights = F.softmax(y, dim=-1)[..., 1]
            max_inds = torch.argsort(weights, descending=True, dim=1)[:, :self.top_k]

            x = torch.gather(x, dim = 1, index=max_inds.unsqueeze(-1).expand(-1,-1,x.shape[-1]))
            y = torch.gather(y, dim = 1, index=max_inds.unsqueeze(-1).expand(-1,-1,y.shape[-1]))

        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=1)
            scan_x = torch.sum(normalized_weights * x, dim=1)
            bt_x = self.bottleneck(scan_x)
            if self.with_fc:
                scan_y = self.fc(bt_x)

        elif self.pool_mode == 'attention':
            weights = self.attention(x)
            normalized_weights = F.softmax(weights, dim=1)
            scan_x = torch.sum(normalized_weights * x, dim=1)
            bt_x = self.bottleneck(scan_x)
            if self.with_fc:
                scan_y = self.fc(bt_x)

        if self.pool_mode == 'score_attention_detach':
            weights = F.softmax(y, dim=2)[..., 1]
            normalized_weights = F.softmax(weights, dim=1).detach().unsqueeze(-1)
            scan_x = torch.sum(normalized_weights * x, dim=1)
            bt_x = self.bottleneck(scan_x)
            if self.with_fc:
                scan_y = self.fc(bt_x)

        # return slice_y, scan_y, weights, bt_x1, scan_x, bt_x

        return scan_y
