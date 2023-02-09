import torch
import torch.nn as nn
import torch.nn.functional as F
# from .torchnets import resnet as resnet
from .mmnets import resnet, resnext
from util.basic import read_multi_data
import torchvision.models as m
backbone_dict = {'resnet': resnet.resnet, 'resnext': resnext.resnext}

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

class MILNet(nn.Module):
    def __init__(self, net_name='resnet', bottleneck_dim=256, class_num=4, dropout_ratio = 0, with_fc = True,
                 pool_mode='gated_attention', top_k = -1, **kwargs):
        super(MILNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = backbone_dict[net_name](pretrained='torchvision://',**kwargs)
        self.pool_mode = pool_mode
        self.top_k = top_k
        self.with_fc = with_fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_ratio = dropout_ratio

        in_features_dict = {50:2048, 34:512, 18:512}
        bottleneck_dim_dict = {50:256, 18:64, 34:256}
        in_features = in_features_dict[kwargs['depth']]
        # self.bottleneck_dim = 256
        self.bottleneck_dim = bottleneck_dim_dict[kwargs['depth']]

        if self.pool_mode in ['attention', 'gated_attention']:
            self.attention_1 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_gate = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_2 = nn.Linear(self.bottleneck_dim, 1)
            
            self.attention_1.apply(init_weights)
            self.attention_gate.apply(init_weights)
            self.attention_2.apply(init_weights)

        # self.bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), torch.nn.Dropout(self.dropout_ratio))
        self.bottleneck = nn.Linear(in_features, self.bottleneck_dim)
        self.fc = nn.Linear(self.bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
            
    def attention(self, x):
        x1 = self.attention_1(x)
        x_out = F.tanh(x1)
        x2 = self.attention_2(x_out)
        return x2

    def gated_attention(self, x):
        x1 = self.attention_1(x)
        x_gate = F.sigmoid(self.attention_gate(x))
        x_out = F.tanh(x1) * x_gate
        x2 = self.attention_2(x_out)
        return x2

    def forward(self, input, len_list = None):
        if len_list is None:
            return self.single_forward(input)
        else:
            y_list = []
            for i in range(input.shape[0]):
                x = input[i][:len_list[i].item()]
                _ ,y, _ = self.single_forward(x)
                y_list.append(y)
            y = torch.cat(y_list, dim = 0)
            return y

    def single_forward(self, input):
        x = self.backbone(input) 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))
        or_x = x
        or_y = y

        if self.top_k > 0:
            weights = F.softmax(y, dim=1)[:, 1]
            max_inds = torch.argsort(weights, descending=True)[:self.top_k]
            x = x[max_inds]
            y = y[max_inds]

        if self.pool_mode == 'score_attention_detach':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = F.softmax(weights, dim=0).detach().view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.bottleneck(total_y)
            if self.with_fc:
                total_y = self.fc(total_y)

        if self.pool_mode == 'score_mean_detach':
            weights = None
            total_y = torch.mean(x, dim=0).unsqueeze(0)
            total_y = self.bottleneck(total_y)
            if self.with_fc:
                total_y = self.fc(total_y)

        if self.pool_mode == 'score_attention':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.bottleneck(total_y)
            if self.with_fc:
                total_y = self.fc(total_y)

        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.bottleneck(total_y)
            if self.with_fc:
                total_y = self.fc(total_y)

        elif self.pool_mode == 'attention':
            weights = self.attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'max':
            scores = F.softmax(y, dim=1)[:, 1]
            max_ind = torch.argmax(scores).item()
            weights = torch.zeros(len(scores), 1)
            weights[max_ind][0] = 1.0
            total_y = self.fc(self.bottleneck(x[max_ind:max_ind+1]))
        elif self.pool_mode == 'clipmean':
            scores = F.softmax(y, dim=1)[:, 1]
            cliplen = 3
            max_score = 0
            max_score_ind = 0
            for i in range(len(scores)-cliplen+1):
                clip_score = torch.sum(scores[i:i+cliplen]).item() / cliplen
                if clip_score > max_score:
                    max_score_ind = i
                    max_score = clip_score
            weights = torch.zeros(len(scores), 1)
            weights[max_score_ind:max_score_ind+cliplen][0] = 1.0 / cliplen
            total_y = torch.sum(weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))

        return or_y, total_y, weights