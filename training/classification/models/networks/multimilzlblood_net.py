from .multimil_net import *

class MultiMILZlbloodNet(MultiMILNet):
    def __init__(self, net_name = 'resnet', class_num=2, with_fc = True,
                 pool_mode='gated_attention', top_k = -1, class_num2 = 2, **kwargs):
        super( MultiMILZlbloodNet, self).__init__(net_name = net_name, class_num=class_num, with_fc = with_fc,
                 pool_mode= pool_mode, top_k = top_k, **kwargs)  # call the initialization method of BaseModel

        in_features = self.in_features

        if self.pool_mode in ['attention', 'gated_attention']:
            self.attention_1_2 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_gate_2 = nn.Linear(in_features, self.bottleneck_dim)
            self.attention_2_2 = nn.Linear(self.bottleneck_dim, 1)

            self.attention_1_2.apply(init_weights)
            self.attention_gate_2.apply(init_weights)
            self.attention_2_2.apply(init_weights)


        self.bottleneck2 = nn.Linear(in_features, self.bottleneck_dim)
        self.fc2 = nn.Linear(self.bottleneck_dim, class_num2)
        self.bottleneck2.apply(init_weights)
        self.fc2.apply(init_weights)

    def gated_attention_2(self, x):
        x1 = self.attention_1_2(x)
        x_gate = F.sigmoid(self.attention_gate_2(x))
        x_out = F.tanh(x1) * x_gate
        x2 = self.attention_2_2(x_out)
        return x2

    def score_attention_detach_forward(self, x, y, atype = 1):
        if atype == 1:
            bottleneck = self.bottleneck
            fc = self.fc
        else:
            bottleneck = self.bottleneck2
            fc = self.fc2

        weights = F.softmax(y, dim=2)[..., 1]
        normalized_weights = F.softmax(weights, dim=1).detach().unsqueeze(-1)
        scan_y = torch.sum(normalized_weights * x, dim=1)
        scan_y = bottleneck(scan_y)
        if self.with_fc:
            scan_y = fc(scan_y)
        return scan_y

    def gated_attention_forward(self, x, atype = 1):
        if atype == 1:
            gated_attention = self.gated_attention
            bottleneck = self.bottleneck
            fc = self.fc
        else:
            gated_attention = self.gated_attention_2
            bottleneck = self.bottleneck2
            fc = self.fc2

        weights = gated_attention(x)
        normalized_weights = F.softmax(weights, dim=1)
        scan_y = torch.sum(normalized_weights * x, dim=1)
        scan_y = bottleneck(scan_y)
        if self.with_fc:
            scan_y = fc(scan_y)
        return scan_y

    def forward(self, input):

        tmp_shape = input.shape
        if len(tmp_shape) == 5:
            n_shape = [tmp_shape[0] * tmp_shape[1], tmp_shape[2], tmp_shape[3], tmp_shape[4]]
            input = input.view(n_shape)

        x = self.backbone(input)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        y = self.fc(self.bottleneck(x))
        y2 = self.fc2(self.bottleneck2(x))

        if len(tmp_shape) != 5:
            return y

        x = x.view(tmp_shape[0], tmp_shape[1], -1)
        y = y.view(tmp_shape[0], tmp_shape[1], -1)
        y2 = y2.view(tmp_shape[0], tmp_shape[1], -1)
        slice_y = y

        if self.top_k > 0:
            weights = F.softmax(y, dim=-1)[..., 1]
            max_inds = torch.argsort(weights, descending=True, dim=1)[:, :self.top_k]

            x = torch.gather(x, dim = 1, index=max_inds.unsqueeze(-1).expand(-1,-1,x.shape[-1]))
            y = torch.gather(y, dim = 1, index=max_inds.unsqueeze(-1).expand(-1,-1,y.shape[-1]))
            y2 = torch.gather(y2, dim = 1, index=max_inds.unsqueeze(-1).expand(-1,-1,y.shape[-1]))

        elif self.pool_mode == 'gated_attention':
            scan_y1 = self.gated_attention_forward(x, atype=1)
            scan_y2 = self.gated_attention_forward(x, atype=2)

        # elif self.pool_mode == 'attention':
        #     weights = self.attention(x)
        #     normalized_weights = F.softmax(weights, dim=1)
        #     scan_y = torch.sum(normalized_weights * x, dim=1)
        #     scan_y = self.bottleneck(scan_y)
        #     if self.with_fc:
        #         scan_y = self.fc(scan_y)
        #
        if self.pool_mode == 'score_attention_detach':
            scan_y1 = self.score_attention_detach_forward(x, y, atype=1)
            scan_y2 = self.score_attention_detach_forward(x, y2, atype=2)
            # weights = F.softmax(y, dim=2)[..., 1]
            # normalized_weights = F.softmax(weights, dim=1).detach().unsqueeze(-1)
            # scan_y = torch.sum(normalized_weights * x, dim=1)
            # scan_y = self.bottleneck(scan_y)
            # if self.with_fc:
            #     scan_y = self.fc(scan_y)

        return slice_y, scan_y1, scan_y2