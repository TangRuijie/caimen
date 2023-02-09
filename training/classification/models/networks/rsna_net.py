import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
# from efficientnet_pytorch import EfficientNet

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

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

class se_resnext101_32x4d(nn.Module):
    def __init__(self):
        super(se_resnext101_32x4d, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
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

class DenseNet121_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)

        return x

class RSNANet(nn.Module):
    def __init__(self, net_name = 'seres', load_pretrained_net = 1, **kwargs):
        super(RSNANet, self).__init__()
        net_dict = {
                    'seres': se_resnext101_32x4d,
                    'dense121': DenseNet121_change_avg,
                    'dense169':DenseNet169_change_avg
        }

        path_dict = {
            'seres': 'buffer/seres_256.pth',
            'dense169':'buffer/dense169_256.pth',
            'dense121':'buffer/dense121_512.pth'
        }

        self.net = net_dict[net_name]()

        if load_pretrained_net:
            print('load pretrained rsna net')
            load_path = path_dict[net_name]
            tmp_net = nn.DataParallel(self.net)
            state_dict = torch.load(load_path, map_location = 'cpu')
            tmp_net.load_state_dict(state_dict['state_dict'], strict = True)
            self.net = tmp_net.module


    def forward(self, x):
        return self.net(x)
