# from torchvision import models
# from .transformer_net import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torchvision.ops import roi_pool
import torch
import math
# from .compact_bilinear_pooling import CompactBilinearPooling
from . import resnet
resnet_dict = {"ResNet18": resnet.resnet18, "ResNet34":resnet.resnet34, "ResNet50":resnet.resnet50, "ResNet101":resnet.resnet101, "ResNet152":resnet.resnet152}
feature_num_dict = {"ResNet18": 512, "ResNet34": 512, "ResNet50": 2048, "ResNet101": 2048, "ResNet152": 2048}

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, batch_size, t, max_seq_len = 80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.b = batch_size
        self.t = t
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        no_time = False
        if len(x.shape) == 4:
            no_time = True
            num_features = x.shape[-3]
            spatial_h = x.shape[-2]
            spatial_w = x.shape[-1]
            x = x.view(self.b, self.t, num_features, spatial_h , spatial_w)
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        spatial_h = x.size(3)
        spatial_w = x.size(4)
        z = Variable(self.pe[:,:seq_len],requires_grad=False)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = z.expand(batch_size, seq_len, num_feature, spatial_h,  spatial_w)
        x = x + z
        if no_time:
            x = x.view(-1, num_features, spatial_h, spatial_w)
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# standard attention layer
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.sum(q.unsqueeze(1) * k, -1)/math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    weights = scores.detach()
    scores = scores.unsqueeze(-1)
    # scores : b, t, dim
    output = scores * v
    output = torch.sum(output,1)
    if dropout:
        output = dropout(output)
    return output, weights

class TX(nn.Module):
    def __init__(self, d_model=64, block_dropout = 0.3, **kwargs):
        super(TX, self).__init__()
        dropout = block_dropout
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff=d_model//2)

    def forward(self, q, k, v, mask=None):
        # b = q.size(0)
        # t = k.size(1)
        # dim = q.size(1)
        # q_temp = q.unsqueeze(1)
        # q_temp= q_temp.expand(b, t, dim)
        A, weights = attention(q, k, v, self.d_model, mask, self.dropout)
        q_ = self.norm_1(A + q)
        new_query = self.norm_2(q_ + self.dropout_2(self.ff(q_)))
        return new_query, weights

class Block_head(nn.Module):
    def __init__(self, d_model = 64, block_layer = 3, head_number = 2, **kwargs):
        super(Block_head, self).__init__()
        self.block_layers = block_layer
        self.head_number = head_number
        self.head_comb_mode = kwargs['head_comb_mode']
        for i in range(head_number):
            for j in range(block_layer):
                # if head_number == 1:
                #     self.add_module('T' + str(i), TX(d_model = d_model, **kwargs))
                # else:
                self.add_module('T' + str(i) + '_' + str(j), TX(d_model = d_model, **kwargs))

    def forward(self, q, k, v, mask=None):
        weights_list = []
        outs = []
        for i in range(self.head_number):
            for j in range(self.block_layers):
                # if self.head_number == 1:
                #     TX = getattr(self, 'T' + str(i))
                # else:
                TX = getattr(self, 'T' + str(i) + '_' + str(j))

                tmp_q, weights = TX(q, k, v)
                weights_list.append(weights)
                q = tmp_q

            outs.append(q)

        out = comb_feature(outs, self.head_comb_mode)

        if len(weights_list) > 0:
            weights = torch.stack(weights_list, dim = -1)
        else:
            weights = None
        return out, weights

class PermuteLayer(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def cal_with_conv(x, r_size, ctype = 'mean'):
    weight = torch.ones(x.shape[1] ,1 , r_size, r_size).cuda(x.device)
    if ctype == 'mean':
        weight /= (r_size**2)
    y = F.conv2d(x, weight, groups = x.shape[1])
    return y

def cal_comb_key(k, v, r_size, layer_m, layer_a):
    '''cal sigma'''

    mu = cal_with_conv(k, r_size, ctype = 'mean')
    mu = mu.permute(0, 2, 3, 1)
    mu = layer_m(mu)
    mu = F.relu(mu)
    key = layer_a(mu).view(mu.shape[0], -1, mu.shape[-1])
    value = cal_with_conv(v, r_size, ctype = 'sum')
    value = value.permute(0, 2, 3, 1)
    value = value.view(value.shape[0], -1, value.shape[-1])

    return key, value

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.3):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.normal(self.linear_1.weight, std=0.001)
        nn.init.normal(self.linear_2.weight, std=0.001)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6, trainable=True):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MyTail(nn.Module):
    def __init__(self,  **kwargs):
        super(MyTail, self).__init__()
        self.spatial_h = kwargs['roi_out_size']
        self.spatial_w = kwargs['roi_out_size']
        self.num_features = kwargs['num_features']
        self.d_model = kwargs['d_model']
        self.crop_size = kwargs['crop_size']
        self.frame_comb_mode = kwargs['frame_comb_mode']
        self.bn1 = nn.BatchNorm2d(self.num_features)
        self.bn2 = Norm(self.d_model, trainable=False)
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn1.bias, 0)

        self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(self.spatial_h, self.spatial_w),
                             stride=1, padding=0, bias=False)

        self.linear_k = nn.Linear(self.num_features, self.d_model)
        self.linear_v = nn.Linear(self.num_features, self.d_model)

        self.block_head = Block_head(**kwargs)
        nn.init.kaiming_normal(self.Qpr.weight, mode='fan_out')

        self.add_conv = 0

        if self.add_conv == 5 or self.add_conv == 6:
            self.layer_m = nn.Linear(self.d_model, self.d_model)
            self.layer_a = nn.Linear(self.d_model, self.d_model)

        if self.add_conv:
            if self.add_conv == 3:
                self.conv_k = nn.Sequential(
                    nn.Conv2d(self.num_features, self.num_features, kernel_size = (self.spatial_h, self.spatial_w), stride = 1),
                    PermuteLayer(),
                    nn.Linear(self.num_features, self.d_model)
                    )
                self.conv_v = nn.Sequential(
                    nn.Conv2d(self.num_features, self.num_features, kernel_size = (self.spatial_h, self.spatial_w), stride = 1),
                    PermuteLayer(),
                    nn.Linear(self.num_features, self.d_model)
                )

            else:
                self.conv_k = nn.Conv2d(self.num_features, self.d_model, kernel_size = (self.spatial_h, self.spatial_w), stride = 1)
                self.conv_v = nn.Conv2d(self.num_features, self.d_model, kernel_size = (self.spatial_h, self.spatial_w), stride = 1)

    def forward(self, x, roi_x, context_range_list):
        """
        :param x:
        :param roi_x:
        :return:
        """

        """step 1: get q from roi_x"""
        roi_x = F.relu(self.Qpr(roi_x))
        roi_x = roi_x.view(-1, self.d_model)
        '''roi out shape: Nx128'''

        # stabilization
        q = roi_x
        n = x.shape[0]

        if self.add_conv == 1 or self.add_conv == 2:
            tmp_k = self.conv_k(x)
            tmp_k = tmp_k.permute(0,2,3,1)
            tmp_k = tmp_k.view(b, -1, self.d_model)
            tmp_v = self.conv_v(x)
            tmp_v = tmp_v.permute(0,2,3,1)
            tmp_v = tmp_v.view(b, -1, self.d_model)

        elif self.add_conv == 3 or self.add_conv == 4:
            tmp_k = self.conv_k(x)
            tmp_k = tmp_k.view(b, -1, self.d_model)
            tmp_v = self.conv_v(x)
            tmp_v = tmp_v.view(b, -1, self.d_model)


        x = x.permute(0,2,3,1)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.add_conv == 5 or self.add_conv == 6:
            tmp_k = k.permute(0,3,1,2)
            tmp_v = v.permute(0,3,1,2)
            tmp_k, tmp_v = cal_comb_key(tmp_k, tmp_v, self.crop_size, layer_m=self.layer_m, layer_a=self.layer_a)

        k = k.view(n, -1, self.d_model)
        v = v.view(n, -1, self.d_model)

        if self.add_conv:
            if self.add_conv%2 == 0:
                k = tmp_k
                v = tmp_v
            else:
                k = torch.cat([k, tmp_k], dim=1)
                v = torch.cat([v, tmp_v], dim=1)

        out_list = []

        last_ind = -1
        tmp_list = []
        for i, context_range in enumerate(context_range_list):
            img_ind, tmp_min, tmp_max = context_range

            if last_ind >= 0 and img_ind != last_ind:
                out = comb_feature(tmp_list, self.frame_comb_mode)
                out_list.append(out)
                tmp_list = []

            last_ind = img_ind

            tmp_q = q[i].unsqueeze(0)
            tmp_k = k[tmp_min: tmp_max + 1]
            tmp_k = tmp_k.view(1, tmp_k.shape[0] * tmp_k.shape[1], tmp_k.shape[2])

            tmp_v = v[tmp_min: tmp_max + 1]
            tmp_v = tmp_v.view(1, tmp_v.shape[0] * tmp_v.shape[1], tmp_v.shape[2])

            out, weight = self.block_head(tmp_q, tmp_k, tmp_v)
            tmp_list.append(out)

        out = comb_feature(tmp_list, self.frame_comb_mode)
        out_list.append(out)
        f = torch.cat(out_list, dim = 0)
        return f
        # F.norma
        # if not self.training:
        #     return f
        # y = self.classifier(f)
        # return y, f

def comb_feature(outs, mode):
    if mode == 'dot':
        out = outs[0]
        for tmp in outs[1:]:
            out *= tmp
    elif mode == 'add':
        out = outs[0]
        for tmp in outs[1:]:
            out += tmp
    elif mode == 'cat':
        out = torch.cat(outs, dim = -1)
    elif mode == 'mean':
        # if len(outs[0].shape) == 1:
        #     outs = [out.unsqueeze(0) for out in outs]
        outs = torch.stack(outs, dim=0)
        out = torch.mean(outs, dim = 0)
    else:
        raise ValueError()
    return out

class ResNet(nn.Module):
    def __init__(self, net_name = 'ResNet50', bottleneck_dim = 256, class_num = 2, with_fc = True, **kwargs):
        super(ResNet,self).__init__()  # call the initialization method of BaseModel

        model_resnet = resnet_dict[net_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        # self.avgpool = model_resnet.avgpool

        batch_size = kwargs['batch_size']
        t = kwargs['seq_len']
        self.embd_pos = kwargs['embd_pos']
        self.heat_layer = kwargs['heat_layer']
        self.up_layer_size = kwargs['up_layer_size']
        self.crop_ratio = kwargs['crop_ratio']
        self.context_frames = kwargs['context_frames']
        self.input_size = kwargs['load_size']
        self.sub_d_model = kwargs['sub_d_model']
        self.roi_out_size = int(self.crop_ratio * self.up_layer_size)

        self.num_features = feature_num_dict[net_name]

        self.pos_embd = PositionalEncoder(self.num_features, max_seq_len=t, batch_size=batch_size, t=t)

        # self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,\
        #                                     self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4)

        self.bottleneck_dim = bottleneck_dim
        self.with_fc = with_fc

        spatial_size = self.up_layer_size * (2**(4 - self.heat_layer))
        self.resize_ratio = self.up_layer_size/self.input_size

        kwargs['roi_out_size'] = self.roi_out_size

        self.tail = MyTail(num_classes = class_num, d_model = self.sub_d_model, crop_size = int(spatial_size * self.crop_ratio),
                           num_features = self.num_features, **kwargs)

        self.roi_op = kwargs['roi_op']
        # if self.roi_op == 'cat':
        #     in_features = 2048 + self.sub_d_model
        # elif self.roi_op == 'add' or self.roi_op == 'mul':
        #     in_features = 2048
        # elif self.roi_op == 'bilinear':
        #     in_features = 16000
        #     self.cbp_layer = CompactBilinearPooling(2048, 2048, 16000)
        # else:
        in_features = self.sub_d_model
        if kwargs['frame_comb_mode'] == 'cat':
            in_features *= kwargs['context_frames']
        if kwargs['head_comb_mode'] == 'cat':
            in_features *= kwargs['head_number']

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

    def forward(self,x, bboxes):
        """
        :param x: B * T * H * W * C
        :param bboxes: tensor B * T * N * [x1 , y1 , x2 , y2]
        :return:
        """
        tmp_shape = x.shape
        if len(tmp_shape) == 5:
            n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
            x = x.view(n_shape)

        for i in range(4 + self.heat_layer):
            x = self.feature_layers[i](x)

        '''
        step 1: construct bboxes        
        '''
        bboxes, context_range_list = self.trans_bboxes(bboxes, self.resize_ratio)
        bboxes = torch.Tensor(bboxes)
        bboxes = bboxes.cuda(device=x.device)

        '''
        step 2: roi bboxes
        '''
        if bboxes.shape[0] == 0:
            roi_x = torch.zeros(tmp_shape[0], self.in_features).cuda(x.device)
        else:
            if self.embd_pos:
                x = self.pos_embd(x)

            roi_x = roi_pool(x, bboxes, [self.roi_out_size, self.roi_out_size], 1)
            # base_x = self.expand_x_with_K(x, bboxes[:, 0, 0, 0, 0].tolist())
            roi_x = self.tail(x, roi_x, context_range_list)
            # roi_x = self.compress_x_with_K(roi_x, bboxes[:, 0, 0, 0, 0].tolist(), x.shape[0])

        x = roi_x

        if self.with_fc:
            if self.bottleneck_dim > 0:
                x = self.bottleneck(x)
            y = self.fc(x)
            return y
        else:
            return x

