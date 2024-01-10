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
device = 'cuda'  



"""
Following code taken from: https://github.com/neuraloperator/graph-pde.git
"""

class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class NNConv_old(torch_geometric.nn.conv.MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.nn)
        size = self.in_channels
        torch_geometric.nn.inits.uniform(size, self.root)
        torch_geometric.nn.inits.uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size, num_edges, edge_index):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]
        self.n_edges = num_edges
        self.edge_index = edge_index

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            else:
                edge_attr = np.zeros((self.n_edges, 3*self.d))
                edge_attr[:,0:2*self.d] = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d +1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            if theta is None:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)



width = 32
ker_width = 256
num_nodes = 1024
depth = 1
edge_features = 4
node_features = 1
L = 100

learning_rate = 0.0001
scheduler_step = 50
scheduler_gamma = 0.8

mynet = KernelNN(width, ker_width, depth, edge_features, node_features).cuda()

optimizer = torch.optim.Adam(mynet.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
edge_index = adj_matrix.nonzero().t().contiguous()
meshgenerator = SquareMeshGenerator([[-L/2, L/2]], [1024], num_nodes*(num_nodes-1), edge_index)

grid = meshgenerator.get_grid()



epochs = 60
batch_size = 50
wavenum_init = 100
lamda_reg = 5

data_train = []
for j in range(trainN):
    edge_attr = meshgenerator.attributes(theta = input_train_torch[:,j])
    data_train.append(torch_geometric.data.Data(x = input_train_torch[:,j], 
                        y = label_train_torch[:,j], edge_index = edge_index, edge_attr = edge_attr))

loss_func = nn.MSELoss()

for ep in range(0, epochs+1):
    for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch, du_label_batch = data_train[indices], label_train_torch[indices], du_label_torch[indices]

        optimizer.zero_grad()
        outputs = step_func(mynet, input_batch, time_step)
        
        loss = loss_func(outputs, label_batch)  # use this loss function for mse loss

        # outputs_2 = step_func(mynet, outputs, time_step) #use these two lines for spectral loss in tendency
        # loss = spectral_loss(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        loss.backward(retain_graph=True)
        
        optimizer.step()


    if ep % 5 == 0:
        print('Epoch', ep)
        print ('Loss', loss)
        # torch.save(mynet.state_dict(), '/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')


torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")