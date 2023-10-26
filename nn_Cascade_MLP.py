import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F


class Cascade_MLP_Net(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int, num_layers: int):
        super(Cascade_MLP_Net, self).__init__()
        self.num_layers = num_layers
        self.il  = ((nn.Linear(input_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.il.weight)      

        self.ol  = nn.Linear(hidden_layer_size,output_size)
        torch.nn.init.xavier_uniform_(self.ol.weight)

        self.layers = nn.ModuleList()
        for i in range(1,num_layers+1): # each hidden layer has size of last + hidden_size
            self.layers.append((nn.Linear(hidden_layer_size*(i), hidden_layer_size)))
            torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.tanh = nn.Tanh()


    def forward(self,x):
        x_state = self.tanh(self.il(x))
        for i in range(0,self.num_layers-1):
            x_out = self.tanh(self.layers[i](x_state))#concatenate each new hidden layer with all previous layers
            x_state = torch.cat(x_state, x_out, dim=-1) #dont do this for last hidden layer
        x_out = self.tanh(self.layers[-1](x_state))
        out =self.ol(x_out)
        return out
