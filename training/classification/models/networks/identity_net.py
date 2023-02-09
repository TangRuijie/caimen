import torch.nn as nn
class IdentityNet(nn.Module):
    def __init__(self,  **kwargs):
        super(IdentityNet, self).__init__()

    def forward(self, x):
        return x
