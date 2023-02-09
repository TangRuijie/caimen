# import torchvision.models as models
from pretrainedmodels.models.torchvision_models import load_pretrained, pretrained_settings
import torch.nn as nn
import types
from ..torchnets import resnet as models
# import torch.utils.model_zoo as model_zoo
# from collections import OrderedDict

# ResNets

def modify_resnets(model, **kwargs):
    # Modify attributs
    model.last_linear = model.fc
    model.fc = None
    # if 'avgpool' in kwargs.keys() and kwargs['avgpool'] is not None:
    #     model.avgpool = kwargs['avgpool']
    # else:
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def resnet18(num_classes=1000, pretrained='imagenet', **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model, **kwargs)
    return model

def resnet34(num_classes=1000, pretrained='imagenet', **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model, **kwargs)
    return model

def resnet50(num_classes=1000, pretrained='imagenet', **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet50(pretrained=False, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model, **kwargs)
    return model

def resnet101(num_classes=1000, pretrained='imagenet', **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model, **kwargs)
    return model

def resnet152(num_classes=1000, pretrained='imagenet', **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False, **kwargs)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model, **kwargs)
    return model

