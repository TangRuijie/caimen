from .resnet import resnet50, resnet18, resnet34, resnet101, resnet152
from .resnext import resnext101_32x4d, resnext101_64x4d
from .senet import se_resnet50, se_resnet101, se_resnet152, senet154, se_resnext101_32x4d, se_resnext50_32x4d
from .densenet import densenet121, densenet169, densenet201, densenet161
from .efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4
from .efficientnet import efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_b8, efficientnet_l2


in_features_dict = {
# 'fbresnet152': 2048,
# 'cafferesnet101': 2048,
# 'bninception': 1024,
'resnext101_32x4d': 2048,
'resnext101_64x4d': 2048,
# 'inceptionv4':1536,
# 'inceptionresnetv2':1536,
# 'nasnetalarge': 2048,
# 'nasnetamobile':1056,
# 'alexnet':4096,
'densenet121':1024,
'densenet169': 1664,
'densenet201': 1920,
'densenet161': 2208,
# 'inceptionv3': 2048,
# 'squeezenet1_0': 2048,
# 'squeezenet1_1': 2048,
# 'vgg11': 4096,
# 'vgg11_bn': 4096,
# 'vgg13': 4096,
# 'vgg13_bn': 4096,
# 'vgg16': 4096,
# 'vgg16_bn': 4096,
# 'vgg19_bn': 4096,
# 'vgg19': 4096,
# 'dpn68': 2048,
# 'dpn68b': 2048,
# 'dpn92': 2048,
# 'dpn98': 2048,
# 'dpn131': 2048,
# 'dpn107': 2048,
'resnet18': 512,
'resnet34': 512,
'resnet50': 2048,
'resnet101': 2048,
'resnet152': 2048,
'xception': 2048,
'senet154': 2048,
'se_resnet50': 2048,
'se_resnet101': 2048,
'se_resnet152': 2048,
'se_resnext50_32x4d': 2048,
'se_resnext101_32x4d': 2048,
# 'pnasnet5large': 2048,
# 'polynet': 2048,
}

