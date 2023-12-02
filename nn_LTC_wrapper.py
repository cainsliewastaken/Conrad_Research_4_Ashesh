import numpy as np
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTC

class LTC_Concat_net(nn.Module):
    def __init__(self, rnn_net, true_x_size: int, hidden_size: int):
        super(LTC_Concat_net, self).__init__()

        self.LTC_net = LTC(true_x_size, hidden_size) #This cannont have mized memory, also it must not have batched inputs or predict multiple timesteps
        self.true_x_size = true_x_size #size of the actuall input/output
        self.init_hidden_state = nn.parameter(torch.rand(hidden_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_true = x[0:self.true_x_size-1]
        hx = x[self.true_x_size:]
        #x is a tensor of size (sequence length (timesteps), num features)
        #hx is a tensor of size (hidden state)
        new_x, hidden_state = self.LTC_net(x_true, hx)
        x = torch.cat(new_x, hidden_state)
        return x