import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F



class MLP_Net(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int):
        super(MLP_Net, self).__init__()
        self.il  = ((nn.Linear(input_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.il.weight)

        self.hidden1  = ((nn.Linear(hidden_layer_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.hidden1.weight)

        self.hidden2  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden2.weight)

        self.hidden3  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden3.weight)

        self.hidden4  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden4.weight)

        self.hidden5  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden5.weight)        

        self.ol  = nn.Linear(hidden_layer_size,output_size)
        torch.nn.init.xavier_uniform_(self.ol.weight)

        self.gelu = F.gelu


    def forward(self,x):
        
        x1 = self.gelu(self.il(x))
        x2 = self.gelu(self.hidden1(x1))
        x3 = self.gelu(self.hidden2(x2))
        x4 = self.gelu(self.hidden3(x3))
        x5 = self.gelu(self.hidden4(x4))
        x6 = self.gelu(self.hidden5(x5))
        out =self.ol(x6)
        return out



class MLP_net_variable(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.gelu, use_act = True):
        super().__init__()
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_in.weight)
        self.activation = activation
        self.layers_1 = nn.ModuleList()
        self.drop = nn.ModuleList()
        self.num_layers = num_layers
        self.use_act = use_act
        for i in range(0,num_layers): 
            self.layers_1.append(nn.Linear(hidden_dim, hidden_dim))
            torch.nn.init.xavier_normal_(self.layers_1[i].weight)
            self.drop.append(nn.Dropout(p=0.9))
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        torch.nn.init.xavier_normal_(self.linear_out.weight)

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        x_0 = x
        for i in range(0,self.num_layers):
            x = self.activation(self.layers_1[i](x)) 
            x = self.drop[i](x)
            # x = x + x_0
        x = self.linear_out(x)
        if self.use_act:
            x = self.activation(x)
        return x