import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from count_trainable_params import count_parameters
import pickle
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss


import torch_geometric
from graph_pde.graph_neural_operator.UAI1_full_resolution import KernelNN
from graph_pde.graph_neural_operator.nn_conv import NNConv_old



lead=1

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/'
 
step_func = Directstep

net_name = 'NN_FNO_Directstep_lead'+str(lead)+'_tendency'

# to changfe from normal loss to spectral loss scroll down 2 right above train for loop

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


time_step = 1e-3
trainN = 150000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).cuda()
du_label_torch = (input_train_torch - label_train_torch).cuda()




width = 32
ker_width = 256
num_nodes = 1024
depth = 1
edge_features = 4
node_features = 1
L = 100
N = 1024
x_vals = np.linspace(-L/2, L/2, N, endpoint=False)


adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
edge_index = adj_matrix.nonzero().t().contiguous()

edge_attr = torch.zeros((num_nodes*(num_nodes-1), 4, trainN))
edge_attr[:, 0, :] = np.repeat(x_vals[edge_index[0,:]], trainN, axis=1) #x
edge_attr[:, 1, :] = np.repeat(x_vals[edge_index[1,:]], trainN, axis=1) #y
edge_attr[:, 2, :] = input_train_torch[edge_index[0,:],:] #f(x)
edge_attr[:, 3, :] = input_train_torch[edge_index[1,:],:] #f(y)


class GNO_net(torch.nn.Module):
    def __init__(self, width, ker_width, depth, edge_size, in_width=1, out_width=1):
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        self.conv1 = GNO_conv(width, width, ker_width, edge_size)



class GNO_conv(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, in_size, out_size, ker_width, edge_size,aggr='mean'):
        super(GNO_conv, self).__init__(aggr=aggr)

        self.in_size = in_size
        self.out_size = out_size
        self.aggr = aggr

        self.ker_1 = torch.nn.Linear(edge_size, ker_width)
        self.ker_2 = torch.nn.Linear(ker_width, ker_width)
        self.ker_3 = torch.nn.Linear(ker_width, width**2)

        self.root = torch.nn.Parameter(torch.Tensor(in_size, out_size))
        self.bias = torch.nn.Parameter(torch.Tensor(out_size))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        hidden_val = self.ker_1(edge_attr)
        hidden_val = F.relu(hidden_val)
        hidden_val = self.ker_2(hidden_val)
        hidden_val = F.relu(hidden_val)
        weight_mat = self.ker_3(hidden_val).view(-1, self.in_size, self.out_size)

        return torch.matmul(x_j, weight_mat)
    
    def update(self, aggr_out, x):
        return aggr_out + torch.mm(x, self.root) + self.bias


model = KernelNN(width, ker_width, depth, edge_features, node_features).cuda()