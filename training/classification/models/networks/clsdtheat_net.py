import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.models import build_detector
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

class ClsDTheatNet(nn.Module):
    def __init__(self, net_name = 'resnet',bottleneck_dim = 256, class_num = 2, with_fc = True, **kwargs):
        super(ClsDTheatNet,self).__init__()  # call the initialization method of BaseModel

        model_resnet = backbone_dict[net_name](pretrained='torchvision://',**kwargs)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.backbone = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,\
                                      self.layer1, self.layer2, self.layer3, self.layer4)

        self.bottleneck_dim = bottleneck_dim
        self.with_fc = with_fc

        detection_checkpoint = kwargs['detection_checkpoint']
        detection_config = kwargs['detection_config']
        self.attention_layers = json.loads(kwargs['attention_layers'])
        self.compose_type = kwargs['compose_type']

        cfg = Config.fromfile(detection_config)
        detection_model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        print("loading dt checkpoint")
        load_checkpoint(detection_model, detection_checkpoint, map_location='cpu')

        self.dt_backbone = detection_model.backbone

        in_features = 2048

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

        if self.compose_type == 'conv':
            for i, e in enumerate(self.attention_layers):
                if e != 0:
                    out_channel = 256
                    for _ in range(0,i):
                        out_channel *= 2

                    conv1x1 = nn.Conv2d(out_channel * 2, out_channel, kernel_size= 1)
                    self.add_module('compose_layer' + str(i), conv1x1)

    def forward(self,input):
        dt_outs = self.dt_backbone(input)
        dt_outs = [out.detach() for out in dt_outs]

        x = input
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i >=4 and i < 8 and self.attention_layers[i - 4]:
                x_attention = dt_outs[i - 4]
                if self.compose_type == 'conv':
                    x = torch.cat((x,x_attention), dim = 1)
                    compose_layer = getattr(self, 'compose_layer' + str(i - 4))
                    x = compose_layer(x)
                else:
                    x = x + x_attention

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)
        y = self.fc(x)
        return y

