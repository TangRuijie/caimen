import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

class LstmNet(nn.Module):
    def __init__(self, class_num = 2 ,embed_size = 2048, LSTM_UNITS = 2048, with_fc = False, DO = 0.3,
                 available_features = [0, 0, 1, 0, 0],  **kwargs):
        super(LstmNet, self).__init__()
        self.available_features = available_features

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        if available_features[2]:
            self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        if available_features[3]:
            self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        if available_features[4]:
            self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.with_fc = with_fc
        if with_fc:
            self.linear = nn.Linear(LSTM_UNITS*2, class_num)

        self.LSTM_UNITS = LSTM_UNITS

    def forward(self, x):

        # h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)
        hidden = torch.zeros(x.shape[0],x.shape[1], self.LSTM_UNITS * 2).cuda(x.device)

        if self.available_features[0]:
            h_embadd = x
            hidden += h_embadd

        h_embedding = x
        h_lstm1, _ = self.lstm1(h_embedding)

        if self.available_features[2]:
            h_lstm2, _ = self.lstm2(h_lstm1)
            hidden += h_lstm2

        if self.available_features[3]:
            h_conc_linear1 = F.relu(self.linear1(h_lstm1))
            hidden += h_conc_linear1

        if self.available_features[4]:
            h_conc_linear2 = F.relu(self.linear2(h_lstm2))
            hidden += h_conc_linear2

        hidden /= len(self.available_features)

        if not self.with_fc:
            return hidden

        y = self.linear(hidden)
        return y

