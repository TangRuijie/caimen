import torch
from . import mynets
import torch.nn as nn
import torch.nn.functional as F
from .utils.weight_init import init_weights
from .layers.identity_layers import IdentityLayer

class MultiMILNet(nn.Module):
    def __init__(self, net_name = 'resnet', class_num=2, with_fc = True,
                 pool_mode='gated_attention', top_k = -1, **kwargs):
        super(MultiMILNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = mynets.__dict__[net_name]()
        in_features = self.backbone.last_linear.in_features

        self.pool_mode = pool_mode
        self.top_k = top_k
        self.with_fc = with_fc

        if 'avgpool_mode' in kwargs.keys() and kwargs['avgpool_mode'] == 'max':
            self.backbone.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.backbone.last_linear = IdentityLayer()

        self.in_features = in_features
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

    def forward(self, input):

        tmp_shape = input.shape
        if len(tmp_shape) == 5:
            n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
            input = input.view(n_shape)

        x = self.backbone(input)
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

        return slice_y, scan_y, weights
