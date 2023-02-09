from ..efficientnet import EfficientNet
import torch.nn.functional as F
import types

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8', 'efficientnet-l2'
)

def modify_efficient(model):
    model.last_linear = model._fc
    del model._fc
    model.avgpool = model._avg_pooling
    del model._avg_pooling

    def features(self, input):
        x = self.extract_features(input)
        return x


    def logits(self, features):
        x = self.avgpool(features)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def efficientnet_b0(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    return modify_efficient(model)

def efficientnet_b1(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b1')
    return modify_efficient(model)

def efficientnet_b2(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b2')
    return modify_efficient(model)

def efficientnet_b3(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b3')
    return modify_efficient(model)

def efficientnet_b4(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    return modify_efficient(model)

def efficientnet_b5(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b5')
    return modify_efficient(model)

def efficientnet_b6(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b6')
    return modify_efficient(model)

def efficientnet_b7(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b7')
    return modify_efficient(model)

def efficientnet_b8(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-b8')
    return modify_efficient(model)

def efficientnet_l2(**kwargs):
    model = EfficientNet.from_pretrained('efficientnet-l2')
    return modify_efficient(model)