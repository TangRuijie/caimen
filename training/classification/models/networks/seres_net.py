import torch.nn as nn
import torch
import torchvision.models
from .cls_net import init_weights
import pretrainedmodels


class SeresNet(nn.Module):
    def __init__(self, class_num = 2, bottleneck_dim = 256, load_pretrained_net = 0, with_fc = True,
                 frozen_layers = 0, dropout_ratio = 0, **kwargs):
        super(SeresNet, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

        self.original_sturcture = kwargs['original_structure']

        self.dropout_ratio = dropout_ratio
        if load_pretrained_net:
            print('load pretrained seres net')
            # load_path = './buffer/model_epoch_best_4.pth'
            load_path = './buffer/seres_256.pth'
            tmp_net = nn.DataParallel(self)
            state_dict = torch.load(load_path, map_location = 'cpu')
            tmp_net.load_state_dict(state_dict['state_dict'], strict = True)
            tmp_net = tmp_net.module
        else:
            tmp_net = self

        if self.original_sturcture:
            return

        self.backbone = nn.Sequential(tmp_net.model_ft.layer0,tmp_net.model_ft.layer1,tmp_net.model_ft.layer2,
                                      tmp_net.model_ft.layer3,tmp_net.model_ft.layer4)

        frozen_layers = max(0, frozen_layers)
        for i in range(frozen_layers):
            m = self.backbone[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        self.bottleneck_dim = bottleneck_dim
        self.with_fc = with_fc

        delattr(self, 'model_ft')

        in_features = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if bottleneck_dim > 0:
            self.bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), torch.nn.Dropout(self.dropout_ratio))
            self.bottleneck.apply(init_weights)

        if with_fc:
            if bottleneck_dim > 0:
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = in_features

    def forward(self,input):
        if self.original_sturcture:
            return self.model_ft(input)

        x = self.backbone(input)
        # x = self.feature_layers(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)

        if self.with_fc:
            y = self.fc(x)
            return y
        else:
            return x
    #
