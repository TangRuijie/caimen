from pretrainedmodels import senet154 as se154
from pretrainedmodels import se_resnet152 as se152
from pretrainedmodels import se_resnet101 as se101
from pretrainedmodels import se_resnet50 as se50
from pretrainedmodels import se_resnext101_32x4d as se101x
from pretrainedmodels import se_resnext50_32x4d as se50x
import torch.nn as nn
import pretrainedmodels as models

def modify_senet(model, **kwargs):
    if 'avg_pool' in kwargs.keys() and kwargs['avg_pool'] is not None:
        model.avg_pool = kwargs['avg_pool']
    else:
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    return model

def se_resnext101_32x4d(pretrained='imagenet', **kwargs):
    model = se101x(pretrained = pretrained)
    return modify_senet(model, **kwargs)

def se_resnext50_32x4d(pretrained='imagenet', **kwargs):
    model = se50x(pretrained = pretrained)
    return modify_senet(model, **kwargs)

def senet154(pretrained='imagenet', **kwargs):
    model = se154(pretrained = pretrained)
    return modify_senet(model, **kwargs)

def se_resnet152(pretrained='imagenet', **kwargs):
    model = se152(pretrained = pretrained)
    return modify_senet(model, **kwargs)

def se_resnet101(pretrained='imagenet', **kwargs):
    model = se101(pretrained = pretrained)
    return modify_senet(model, **kwargs)

def se_resnet50(pretrained='imagenet', **kwargs):
    model = se50(pretrained = pretrained)
    return modify_senet(model, **kwargs)
