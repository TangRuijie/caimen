import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from mmdet.models import build_detector
from mmcv import Config
from mmcv.runner import get_dist_info, load_checkpoint
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

class MILNet(nn.Module):

    def __init__(self, net_name='resnet', bottleneck_dim=256, class_num=4, pool_mode='gated_attention', **kwargs):
        super(MILNet,self).__init__()  # call the initialization method of BaseModel

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

        self.backbone = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                      self.layer1, self.layer2, self.layer3, self.layer4)
        self.pool_mode = pool_mode
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features_dict = {50:2048, 18:512}
        # bottleneck_dim_dict = {50:256, 18:64}
        bottleneck_dim_dict = {50:256, 18:256}
        in_features = in_features_dict[kwargs['depth']]
        # self.bottleneck_dim = 256
        self.bottleneck_dim = bottleneck_dim_dict[kwargs['depth']]

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

    def dt_forward(self, input):
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

        return x

    def forward(self, input):
        # x = self.backbone(input)
        x = self.dt_forward(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))

        if self.pool_mode == 'score_attention_detach':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = F.softmax(weights, dim=0).detach().view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        if self.pool_mode == 'score_attention':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'attention':
            weights = self.attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'max':
            scores = F.softmax(y, dim=1)[:, 1]
            max_ind = torch.argmax(scores).item()
            weights = torch.zeros(len(scores), 1)
            weights[max_ind][0] = 1.0
            total_y = self.fc(self.bottleneck(x[max_ind:max_ind+1]))
        elif self.pool_mode == 'clipmean':
            scores = F.softmax(y, dim=1)[:, 1]
            cliplen = 3
            max_score = 0
            max_score_ind = 0
            for i in range(len(scores)-cliplen+1):
                clip_score = torch.sum(scores[i:i+cliplen]).item() / cliplen
                if clip_score > max_score:
                    max_score_ind = i
                    max_score = clip_score
            weights = torch.zeros(len(scores), 1)
            weights[max_score_ind:max_score_ind+cliplen][0] = 1.0 / cliplen
            total_y = torch.sum(weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))

        return y, total_y, weights