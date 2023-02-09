import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
import torch.nn as nn
from mmdet.models import build_detector
from mmcv.runner import get_dist_info, load_checkpoint
from mmcv import Config

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

class ResHeatNet50(nn.Module):
    def __init__(self, detection_config, detection_checkpoint, out_channel,use_bottleneck, bottleneck_dim, use_or_feature = True):
        super(ResHeatNet50, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.use_or_heat = use_or_feature
        self.bottleneck_dim = bottleneck_dim
        self.out_channel = out_channel
        cfg = Config.fromfile(detection_config)
        detection_model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        
        if detection_checkpoint is not None:
            load_checkpoint(detection_model, detection_checkpoint, map_location='cpu')
        self.feature_layers = detection_model.backbone

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        fc_dim = 2048
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(fc_dim, bottleneck_dim)
            self.bottleneck.apply(init_weights)
            fc_dim = bottleneck_dim

        self.fc = nn.Linear(fc_dim, out_channel)
        self.fc.apply(init_weights)

    def forward(self, x, heatmap):
        x = x.float()
        outs = list(self.feature_layers(x))
        x = outs[-1]
        heatmap = F.interpolate(heatmap, (x.shape[2], x.shape[3]), mode='bilinear')
        x_attention = x * heatmap
        if self.use_or_heat:
            x = torch.cat((x,x_attention), dim = 1)
        else:
            x = x_attention

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.fc(x)

        return x
