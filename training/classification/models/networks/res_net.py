from torchvision import models
import torch.nn as nn
from .torchnets import resnet as resnet
from .attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0

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

class ResNet(nn.Module):

    def __init__(self, net_name = 'ResNet50', bottleneck_dim = 256, class_num = 2, norm_method = 'batch', **kwargs):
        super(ResNet,self).__init__()  # call the initialization method of BaseModel

        # model_resnet = resnet_dict[net_name](pretrained=True, norm_type = norm_method)
        model_resnet = resnet_dict[net_name](pretrained=True, norm_type = norm_method)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.layer_list = [self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4]

        self.bottleneck_dim = bottleneck_dim
        if self.bottleneck_dim > 0:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim

    def forward(self,input):
        x = self.backbone(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.with_fc:
            if self.bottleneck_dim > 0:
                x = self.bottleneck(x)
            y = self.fc(x)
            return y
        else:
            return x

class ResAttentionNet(nn.Module):

    def __init__(self, net_name = 'ResNet50', m_scale = 1, clip_len = 5, use_bottleneck = True, new_cls = True,
                 bottleneck_dim = 256, class_num = 4, with_fc = True):
        super(ResAttentionNet,self).__init__()  # call the initialization method of BaseModel

        print('attention net')

        model_resnet = resnet_dict[net_name](pretrained=True)
        if clip_len == 3:
            self.conv1 = model_resnet.conv1
        else:
            self.conv1 = nn.Conv2d(clip_len * m_scale, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.attention1 = AttentionModule_stage1(256, 256)
        self.layer2 = model_resnet.layer2
        self.attention2 = AttentionModule_stage2(512, 512)
        self.attention2_2 = AttentionModule_stage2(512, 512)
        self.layer3 = model_resnet.layer3
        self.attention3 = AttentionModule_stage3(1024, 1024)
        self.attention3_2 = AttentionModule_stage3(1024, 1024)
        self.attention3_3 = AttentionModule_stage3(1024, 1024)
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        # self.layer_list = [self.conv1, self.bn1, self.relu, self.maxpool, \
        #                                     self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]

        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.attention1,
                                            self.layer2, self.attention2,self.attention2_2,
                                            self.layer3, self.attention3,self.attention3_2,self.attention3_3,
                                            self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.with_fc = with_fc

        if with_fc:
            if self.new_cls:
                if self.use_bottleneck:
                    self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                    self.fc = nn.Linear(bottleneck_dim, class_num)
                    self.bottleneck.apply(init_weights)
                    self.fc.apply(init_weights)
                    self.__in_features = bottleneck_dim
                else:
                    self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                    self.fc.apply(init_weights)
                    self.__in_features = model_resnet.fc.in_features
            else:
                self.fc = model_resnet.fc
                self.__in_features = model_resnet.fc.in_features

    def forward(self,input):
        x = self.feature_layers(input)
        x = x.view(x.size(0), -1)
        if self.with_fc:
            if self.use_bottleneck and self.new_cls:
                x = self.bottleneck(x)
            y = self.fc(x)
            return y
        else:
            return x


# base_network = ResNet()
# base_network = base_network.cuda()
# tmp_input = torch.Tensor(3,5,224,224).cuda()
# c = base_network(tmp_input)
#
# print('success')
