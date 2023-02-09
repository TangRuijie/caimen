from torchvision import models
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from .torchnets import resnet as resnet
# from .attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0

# resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
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

class ResHeatNet(nn.Module):
    def __init__(self, net_name = 'ResNet50', bottleneck_dim = 256, class_num = 2, **kwargs):
        super(ResHeatNet,self).__init__()  # call the initialization method of BaseModel
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

        self.attention_layers = json.loads(kwargs['attention_layers'])
        self.compose_type = kwargs['compose_type']
        self.bottleneck_dim = bottleneck_dim

        if self.compose_type == 'conv':
            self.compose_layers = []
            for i, e in enumerate(self.attention_layers):
                if e != 0:
                    out_channel = 256
                    for _ in range(0,i):
                        out_channel *= 2

                    conv1x1 = nn.Conv2d(out_channel * 2, out_channel, kernel_size= 1)
                    self.add_module('compose' + str(i), conv1x1)
                else:
                    conv1x1 = None
                self.compose_layers.append(conv1x1)

        self.layer_list = [self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]

        self.backbone = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4)

        if bottleneck_dim > 0:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)

        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features

    def forward(self,input, heatmap):
        x = input
        for i, layer in enumerate(self.layer_list):
            x = layer(x)
            if i >=4 and i < 8 and self.attention_layers[i - 4]:
                heatmap = F.interpolate(heatmap, (x.shape[2], x.shape[3]), mode='bilinear')
                x_attention = x * heatmap

                if self.compose_type == 'conv':
                    x = torch.cat((x,x_attention), dim = 1)
                    x = self.compose_layers[i - 4](x)
                else:
                    x = x + x_attention

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)
        y = self.fc(x)
        return y

