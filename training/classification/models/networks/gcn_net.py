import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.gcn_layers import GraphConvolution


class GCNNet(nn.Module):
    def __init__(self, nfeat, middle_feat, nclass = 2, dropout_rate=0.6, with_fc = False):
        super(GCNNet, self).__init__()
        # original layers
        # self.fc1 = nn.Linear(nfeat, 512)
        # self.fc2 = nn.Linear(512, 128)
        # Graph Convolution

        nfeat2 = middle_feat

        self.with_fc = with_fc

        self.gc1 = GraphConvolution(nfeat, nfeat2) #nn.Linear(128, 32)
        self.gc3 = GraphConvolution(nfeat, nfeat2) #nn.Linear(128, 32)

        if self.with_fc:
            self.gc2 = GraphConvolution(nfeat2, nclass)
            self.gc4 = GraphConvolution(nfeat2, nclass) #nn.Linear(128, 32)

        self.dropout_rate = dropout_rate

    def forward(self, x):
        adj = torch.zeros(x.shape[1], x.shape[1])
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                adj[i, j] = torch.exp(-torch.abs(torch.tensor(i-j).float()))

        adj = adj.cuda(x.device)
        assert (x.shape[0] == 1)
        # original layers
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, self.dropout_rate, training=self.training)

        # Graph Convolution
        x1 = F.relu(self.gc1(x, adj))
        if self.with_fc:
            x1 = self.gc2(x1, adj)

        # Learnable Graph branch
        x2 = x.view(-1, x.shape[-1])
        x_norm = torch.norm(x2, p=2, dim=1)
        x_norm = x_norm.view(-1, 1)
        x2 = x2.matmul(x2.t())
        x_norm = x_norm.matmul(x_norm.t())
        adj2 = torch.exp(x2 - x2.max(dim=1,keepdim=True)[0])#1 + x2 / x_norm
        d_inv_sqrt2 = torch.diag(torch.pow(torch.sum(adj2, dim=1), -0.5))
        adj_hat2 = d_inv_sqrt2.matmul(adj2).matmul(d_inv_sqrt2)
        adj_hat2 = adj_hat2.view(x.shape[0], adj_hat2.shape[0], adj_hat2.shape[1])

        y2 = F.relu(self.gc3(x, adj_hat2))

        if self.with_fc:
            y22 = y2.view(-1, y2.shape[-1])
            y2_norm = torch.norm(y22, p=2, dim=1).view(-1, 1)
            y22 = y22.matmul(y22.t())
            y2_norm = y2_norm.matmul(y2_norm.t())
            adj3 = 1 + y22 / y2_norm
            d_inv_sqrt3 = torch.diag(torch.pow(torch.sum(adj3, dim=1), -0.5))
            adj_hat3 = d_inv_sqrt3.matmul(adj3).matmul(d_inv_sqrt3)
            adj_hat3 = adj_hat3.view(x.shape[0], adj_hat3.shape[0], adj_hat3.shape[1])
            y2 = self.gc4(y2, adj_hat3)

        return (x1 + y2) / 2.0