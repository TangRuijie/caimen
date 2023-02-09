import torch.nn as nn
from .cls_net import ClsNet

class ClsCleanerNet(ClsNet):
    def __init__(self, **kwargs):
        super(ClsCleanerNet, self).__init__(**kwargs)
        self.use_bt_feature = kwargs['use_bt_feature']

    def feature_forward(self,input):
        x = self.backbone(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.bottleneck_dim > 0 and self.use_bt_feature:
            x = self.bottleneck(x)

        return x

    def label_forward(self,input):
        x = input
        if self.bottleneck_dim > 0 and not self.use_bt_feature:
            x = self.bottleneck(x)
        y = self.fc(x)
        return y


    def full_forward(self,input):
        x = self.backbone(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)

        y = self.fc(x)
        return y

    def forward(self,input, ftype = 0):
        if ftype == 1:
            return self.feature_forward(input)
        if ftype == 2:
            return self.label_forward(input)

        return self.full_forward(input)

