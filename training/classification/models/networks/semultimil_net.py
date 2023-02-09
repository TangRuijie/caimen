import torch
import torch.nn as nn
import torch.nn.functional as F
# from .torchnets import resnet as resnet
# from .mmnets import resnet, resnext
import pretrainedmodels
# backbone_dict = {'resnet': resnet.resnet, 'resnext': resnext.resnext}

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

class SeMultiMILNet(nn.Module):
    def __init__(self, class_num = 2, bottleneck_dim = 256, load_pretrained_net = 0, with_fc = True,top_k = -1,
                 frozen_layers = 0, dropout_ratio = 0, pool_mode = ''):
        super(SeMultiMILNet,self).__init__()  # call the initialization method of BaseModel
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        self.pool_mode = pool_mode
        # num_ftrs = self.model_ft.last_linear.in_features
        # self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

        self.dropout_ratio = dropout_ratio
        if load_pretrained_net:
            load_path = './buffer/model_epoch_best_4.pth'
            tmp_net = nn.DataParallel(self)
            state_dict = torch.load(load_path, map_location = 'cpu')
            tmp_net.load_state_dict(state_dict['state_dict'], strict = False)
            tmp_net = tmp_net.module
            print('load pretrained seres net')
        else:
            tmp_net = self

        self.backbone = nn.Sequential(tmp_net.model_ft.layer0,tmp_net.model_ft.layer1,tmp_net.model_ft.layer2,
                                      tmp_net.model_ft.layer3,tmp_net.model_ft.layer4)

        if frozen_layers < 0:
            frozen_layers = 0

        for i in range(frozen_layers):
            m = self.backbone[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        self.bottleneck_dim = bottleneck_dim
        self.with_fc = with_fc

        delattr(self, 'model_ft')

        in_features = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        if self.pool_mode in ['attention', 'gated_attention']:
            self.attention_1 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_gate = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_2 = nn.Linear(self.bottleneck_dim, 1)
            
            self.attention_1.apply(init_weights)
            self.attention_gate.apply(init_weights)
            self.attention_2.apply(init_weights)

        self.bottleneck = nn.Sequential(nn.Linear(in_features, bottleneck_dim), torch.nn.Dropout(self.dropout_ratio))
        # self.bottleneck = nn.Linear(in_features, self.bottleneck_dim)
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

    def forward(self, input):
        tmp_shape = input.shape
        n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
        input = input.view(n_shape)
        x = self.backbone(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))

        x = x.view(tmp_shape[0], tmp_shape[1], -1)
        y = y.view(tmp_shape[0], tmp_shape[1], -1)

        or_x = x
        or_y = y

        # if self.top_k > 0:
        #     weights = F.softmax(y, dim=1)[:, 1]
        #     max_inds = torch.argsort(weights, descending=True)[:self.top_k]
        #     x = x[max_inds]
        #     y = y[max_inds]

        if self.pool_mode == 'score_attention_detach':
            weights = F.softmax(y, dim=2)[..., 1]
            normalized_weights = F.softmax(weights, dim=1).detach().unsqueeze(-1)
            total_y = torch.sum(normalized_weights * x, dim=1)
            total_y = self.bottleneck(total_y)
            if self.with_fc:
                total_y = self.fc(total_y)

        elif self.pool_mode == 'gated_attention':
            weights = self.gated_attention(x)
            normalized_weights = F.softmax(weights, dim=1)
            total_y = torch.sum(normalized_weights * x, dim=1)
            total_y = self.bottleneck(total_y)
            if self.with_fc:
                total_y = self.fc(total_y)

        elif self.pool_mode == 'max':
            scores = F.softmax(y, dim=2)[..., 1]
            max_inds = torch.argmax(scores, dim = -1)
            weights = None
            # weights[max_ind][0] = 1.0
            tmp_x = []
            for i in range(max_inds.shape[0]):
                tmp_x.append(x[i,max_inds[i]])
            total_y = torch.stack(tmp_x)

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

        elif self.pool_mode == 'attention':
            weights = self.attention(x)
            normalized_weights = F.softmax(weights, dim=0).view(-1, 1)
            total_y = torch.sum(normalized_weights * x, dim=0).unsqueeze(0)
            total_y = self.fc(self.bottleneck(total_y))
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