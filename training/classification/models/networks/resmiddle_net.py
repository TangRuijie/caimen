from torchvision import models
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from .torchnets import resnet as resnet
import copy

resnet_dict = {"ResNet18": resnet.resnet18, "ResNet34":resnet.resnet34, "ResNet50":resnet.resnet50, "ResNet101":resnet.resnet101, "ResNet152":resnet.resnet152}

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

class ResMiddleNet(nn.Module):
    def __init__(self, net_name = 'ResNet50', bottleneck_dim = 256, class_num = 2, **kwargs):
        super(ResMiddleNet,self).__init__()  # call the initialization method of BaseModel
        model_resnet = resnet_dict[net_name](pretrained=True , **kwargs)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.copy_start_layer = kwargs['copy_start_layer']
        self.share_layers = kwargs['share_layers']
        self.bottleneck_dim = bottleneck_dim
        self.fc_num = 4096

        self.backbone = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4)

        if not self.share_layers:
            # self.copy_layers = self.layer_list[self.copy_start_layer + 3: ]
            copy_layers = []
            for i, layer in enumerate(self.backbone[self.copy_start_layer + 3: ]):
                c_layer = copy.deepcopy(layer)
                copy_layers.append(c_layer)
            self.copybone = nn.Sequential(*copy_layers)

        if bottleneck_dim > 0:
            self.bottleneck = nn.Linear(self.fc_num, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)

        else:
            self.fc = nn.Linear(self.fc_num, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features

    def forward(self,input, heatmap):
        x = input
        heat = None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i >= 4 and i - 3 >= self.copy_start_layer:
                if self.share_layers:
                    heat = layer(heat)
                else:
                    heat = self.copybone[i - 3 - self.copy_start_layer](heat)

            if i - 3 == self.copy_start_layer - 1:
                heatmap = F.interpolate(heatmap, (x.shape[2], x.shape[3]), mode='bilinear')
                heat = x * heatmap

        x = torch.cat([x, heat], dim = 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)
        y = self.fc(x)
        return y

