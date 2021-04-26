import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import params

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size,layers=1):
        super(LSTM, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.layers=layers
        self.lstm=nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=layers)

        self.hidden2tag = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(self.layers,params.batch_size, self.hidden_dim).cuda(),
                torch.randn(self.layers, params.batch_size, self.hidden_dim).cuda())

    def forward(self, input):
        lstm_out,self.hidden = self.lstm(
            input,self.hidden)
        tag_space = self.hidden2tag(lstm_out[-1,:,:])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


