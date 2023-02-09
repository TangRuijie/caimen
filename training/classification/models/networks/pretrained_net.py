import torch.nn as nn
import torchvision
from .utils import weight_init

import pretrainedmodels

class PretrainedNet(nn.Module):

    net_name_list = ['densenet169', 'densenet121']
    in_features_dict = {
        'densenet169': 1664,
    }

    def __init__(self, net_name):
        super(PretrainedNet, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x

class DenseNet169_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet169(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x


class se_resnext50_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext50_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x

