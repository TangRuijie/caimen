from . import mynets
import torch.nn as nn
from .utils.weight_init import init_weights
from .layers.identity_layers import IdentityLayer

class ClsNet(nn.Module):
    def __init__(self, net_name = 'resnet18', class_num = 4, bottleneck_dim = 256, with_fc = True,
                 avgpool = None, last_linear = None, pretrained = 'imagenet', **kwargs):
        super(ClsNet, self).__init__()  # call the initialization method of BaseModel
        self.backbone = mynets.__dict__[net_name](pretrained = pretrained)
        self.with_fc = with_fc
        in_features = self.backbone.last_linear.in_features

        if avgpool is not None:
            self.backbone.avgpool = avgpool

        if last_linear is not None:
            self.backbone.last_linear = last_linear
        else:
            self.backbone.last_linear = IdentityLayer()

        if with_fc:
            if bottleneck_dim > 0:
                bottleneck = nn.Linear(in_features, bottleneck_dim)
                fc = nn.Linear(bottleneck_dim, class_num)
                bottleneck.apply(init_weights)
                fc.apply(init_weights)
                self.fc = nn.Sequential(bottleneck, fc)
            else:
                fc = nn.Linear(in_features, class_num)
                fc.apply(init_weights)
                self.fc = nn.Sequential(in_features, fc)

    def forward(self, input):
        x = self.backbone(input)
        if self.with_fc:
            x = self.fc(x)
        return x
