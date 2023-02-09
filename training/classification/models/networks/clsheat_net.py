import torch
import torch.nn as nn
import torch.nn.functional as F
# from .torchnets import resnet as resnet
from .mmnets import resnet, resnext
import torchvision.models as m
backbone_dict = {'resnet': resnet.resnet, 'resnext': resnext.resnext}

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

class ClsheatNet(nn.Module):
    def __init__(self, net_name = 'resnet',bottleneck_dim = 256, class_num = 4, with_fc = True, **kwargs):
        super(ClsheatNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = backbone_dict[net_name](pretrained='torchvision://',**kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.with_fc = with_fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 2048 * 2

        if with_fc:
            if bottleneck_dim > 0:
                self.bottleneck = nn.Linear(in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = in_features


    def forward(self,input, heatmap):
        x = self.backbone(input)
        heatmap = F.interpolate(heatmap, (x.shape[2], x.shape[3]), mode='bilinear')
        x_attention = x * heatmap
        x = torch.cat((x,x_attention), dim = 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.with_fc:
            if self.bottleneck_dim > 0:
                x = self.bottleneck(x)
            y = self.fc(x)
            return y
        else:
            return x
