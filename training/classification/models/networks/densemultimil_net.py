import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .utils.weight_init import init_weights
from .layers.identity_layers import IdentityLayer
import pretrainedmodels

class MultiMILNet(nn.Module):
    def __init__(self, net_name = 'resnet', class_num=2, with_fc = True,
                 pool_mode='gated_attention', top_k = -1, **kwargs):
        super(MultiMILNet,self).__init__()  # call the initialization method of BaseModel

        super(MultiMILNet, self).__init__()
        self.backbone = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        # self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()
        self.pool_mode = pool_mode
        self.top_k = top_k
        self.with_fc = with_fc
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 1024
        self.bottleneck_dim = 256

        if kwargs['target_focus'] == 2:
            self.bottleneck_dim = 64

        if self.pool_mode in ['attention', 'gated_attention']:
            self.attention_1 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_gate = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_2 = nn.Linear(self.bottleneck_dim, 1)
            
            self.attention_1.apply(init_weights)
            self.attention_gate.apply(init_weights)
            self.attention_2.apply(init_weights)

        # self.bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), torch.nn.Dropout(self.dropout_ratio))
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

        # x = self.avgpool(x)
        # x = x.view(-1, 1024)
        # x = self.mlp(x)

        x = self.backbone(input)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 1024)
        y = self.fc(self.bottleneck(x))

        if len(tmp_shape) != 5:
            return y

        x = x.view(tmp_shape[0], tmp_shape[1], -1)
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
            scan_y = torch.sum(normalized_weights * x, dim=1)
            scan_y = self.bottleneck(scan_y)
            if self.with_fc:
                scan_y = self.fc(scan_y)

        elif self.pool_mode == 'attention':
            weights = self.attention(x)
            normalized_weights = F.softmax(weights, dim=1)
            scan_y = torch.sum(normalized_weights * x, dim=1)
            scan_y = self.bottleneck(scan_y)
            if self.with_fc:
                scan_y = self.fc(scan_y)

        if self.pool_mode == 'score_attention_detach':
            weights = F.softmax(y, dim=2)[..., 1]
            normalized_weights = F.softmax(weights, dim=1).detach().unsqueeze(-1)
            scan_y = torch.sum(normalized_weights * x, dim=1)
            scan_y = self.bottleneck(scan_y)
            if self.with_fc:
                scan_y = self.fc(scan_y)

        return slice_y, scan_y, weights