import torchvision.models as models
from pretrainedmodels.models.torchvision_models import load_pretrained, pretrained_settings
import torch.nn.functional as F
import torch.nn as nn
import types


# DenseNets

def modify_densenets(model, **kwargs):
    # Modify attributs
    model.last_linear = model.classifier
    # if 'avgpool' in kwargs.keys() and kwargs['avgpool'] is not None:
    #     model.avgpool = kwargs['avgpool']
    # else:
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        # x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def densenet121(num_classes=1000, pretrained='imagenet', **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet121'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model, **kwargs)
    return model

def densenet169(num_classes=1000, pretrained='imagenet', **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet169'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model, **kwargs)
    return model

def densenet201(num_classes=1000, pretrained='imagenet', **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet201'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model, **kwargs)
    return model

def densenet161(num_classes=1000, pretrained='imagenet', **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet161'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model, **kwargs)
    return model
