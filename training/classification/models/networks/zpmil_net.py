import torch
import torch.nn as nn
import torch.nn.functional as F
# from .torchnets import resnet as resnet
from .mmnets import resnet, resnext
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

    def __init__(self, net_name='resnet', bottleneck_dim=256, class_num=4, pool_mode='score_attention_detach', **kwargs):
        super(MILNet,self).__init__()  # call the initialization method of BaseModel

        self.backbone = backbone_dict[net_name](pretrained='torchvision://',**kwargs)
        self.pool_mode = pool_mode
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features_dict = {50:2048, 18:512}
        bottleneck_dim_dict = {50:256, 18:64}
        # bottleneck_dim_dict = {50:256, 18:256}
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

    def forward(self, input, cliplen=3, topk=3):
        x = self.backbone(input) 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))

        if self.pool_mode == 'attention':
            weights = self.attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'exp_score_attn':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = F.softmax(weights, dim=0).detach().view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'exp_score':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.FloatTensor((1,2), device=y.device, requires_grad=True)
            total_y[0][1] = torch.sum(normalized_weights * y[:, 1])
            normalized_weights = normalized_weights.detach()
            total_y[0][0] = torch.sum(normalized_weights * y[:, 0])
        elif self.pool_mode == 'linear_score_attn':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = weights / sum(weights)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'linear_score':
            weights = F.softmax(y, dim=1)[:, 1]
            normalized_weights = weights / sum(weights)
            total_y = torch.FloatTensor((1,2), device=y.device, requires_grad=True)
            total_y[0][1] = torch.sum(normalized_weights * y[:, 1])
            normalized_weights = normalized_weights.detach()
            total_y[0][0] = torch.sum(normalized_weights * y[:, 0])
        elif self.pool_mode == 'max':
            scores = F.softmax(y, dim=1)[:, 1]
            max_ind = torch.argmax(scores).item()
            weights = torch.zeros(len(scores), 1)
            weights[max_ind][0] = 1.0
            total_y = self.fc(self.bottleneck(x[max_ind:max_ind+1]))
        elif self.pool_mode == 'average':
            weights = torch.zeros(y.shape[0], 1)
            weights[:] = 1.0 / y.shape[0]
            total_y = torch.sum(x, dim=0).unsqueeze(0) / y.shape[0]
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'topkmean':
            scores = F.softmax(y, dim=1)[:, 1]
            _, max_score_inds = scores.topk(topk)
            weights = torch.zeros(len(scores), 1)
            for ind in max_score_inds:
                weights[int(ind)][0] = 1.0 / topk
            weights = weights.cuda(x.device)
            total_y = torch.sum(weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'clipmax':
            scores = F.softmax(y, dim=1)[:, 1]
            max_score = 0
            max_score_ind = 0
            for i in range(len(scores)-cliplen+1):
                clip_score = torch.sum(scores[i:i+cliplen]).item() / cliplen
                if clip_score > max_score:
                    max_score_ind = i
                    max_score = clip_score
            weights = torch.zeros(len(scores), 1)
            weights[max_score_ind:max_score_ind+cliplen][0] = 1.0 / cliplen
            weights = weights.cuda(x.device)
            total_y = torch.sum(weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
        elif self.pool_mode == 'TKS':
            scores = F.softmax(y, dim=1)[:, 1]
            seg_scores = torch.FloatTensor(len(scores)-cliplen+1)
            max_score = 0
            max_score_ind = 0
            for i in range(len(scores)-cliplen+1):
                seg_scores[i] = torch.sum(scores[i:i+cliplen]) / cliplen
            _, max_score_inds = seg_scores.topk(topk)
            weights = torch.zeros(len(scores), 1)
            for ind in max_score_inds:
                weights[int(ind):int(ind)+cliplen, 0] += 1.0 / (topk * cliplen)
            weights = weights.cuda(x.device)
            total_y = torch.sum(weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))

        return y, total_y, weights