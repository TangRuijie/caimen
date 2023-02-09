from pretrainedmodels import resnext101_32x4d as res32x4d
from pretrainedmodels import resnext101_64x4d as res64x4d
import torch.nn as nn
import pretrainedmodels as models

def modify_resx4d(model, **kwargs):
    if 'avg_pool' in kwargs.keys() and kwargs['avg_pool'] is not None:
        model.avg_pool = kwargs['avg_pool']
    else:
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    return model

def resnext101_32x4d(pretrained='imagenet', **kwargs):
    model = res32x4d(pretrained = pretrained)
    return modify_resx4d(model, **kwargs)

def resnext101_64x4d(pretrained='imagenet', **kwargs):
    model = res64x4d(pretrained = pretrained)
    return modify_resx4d(model, **kwargs)