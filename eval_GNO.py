import numpy as np
import torch
import scipy.io
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from count_trainable_params import count_parameters
import pickle
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss
import torch_geometric
from sklearn.metrics import pairwise_distances 


time_step = 1e-3
lead = int((1/1e-3)*time_step)

skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_GNO/' #this is where the saved graphs and .mat files end up

net_file_path = "/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/NN_GNO_PEC4step_lead1.pt" #change this to use a different network

step_func = PEC4step #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

eval_output_name = 'GNO_predicted_PEC4step_1024_lead'+str(lead)+''  # what to name the output file, .mat ending not needed

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f: #change for eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:-1:lead])

print(label_test.shape)

device = 'cuda'  


"""
Following code taken from: https://github.com/neuraloperator/graph-pde.git
"""

class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width, out_width, edge_attr, edge_index):
        super(KernelNN, self).__init__()
        self.depth = depth
        self.edge_attr = edge_attr.cuda()
        self.edge_index = edge_index

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, out_width)

    def forward(self, x):
        x = x.unsqueeze(-1)
        edge_attr = torch.zeros((self.edge_index.shape[1], 4)).cuda()
        edge_attr[:,0:2] = self.edge_attr
        edge_attr[:,2] = x[self.edge_index[0]].squeeze(-1)
        edge_attr[:,3] = x[self.edge_index[1]].squeeze(-1)
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, self.edge_index, edge_attr))
        x = self.fc2(x).squeeze(-1)
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
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = torch.tensor(np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1)))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = torch.tensor(np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T)

    def ball_connectivity(self, r):
        pwd = pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long).cuda()
    

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            else:
                edge_attr = torch.zeros((self.n_edges, 4))
                edge_attr[:,0:2*self.d] = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]].squeeze(-1)
                edge_attr[:, 2 * self.d +1] = theta[self.edge_index[1]].squeeze(-1)
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            if theta is None:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return edge_attr
    
width = 16
ker_width = 128
num_nodes = 1024
depth = 4
edge_features = 4
edge_radius = 3
node_features = 1
L = 100


# adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes) #define graph edges
# edge_index = adj_matrix.nonzero().t().contiguous().cuda()
meshgenerator = SquareMeshGenerator([[-1, 1]], [1024]) #define function to find graph edges
edge_index = meshgenerator.ball_connectivity(edge_radius/50)
edge_attr = meshgenerator.attributes()



mynet = KernelNN(width, ker_width, depth, edge_features, node_features, node_features, edge_attr, edge_index).cuda()
mynet.load_state_dict(torch.load(net_file_path))
mynet.cuda()
print('Model loaded')

M = int(np.floor(99999/lead))
net_pred = np.zeros([M,np.size(label_test,1)])


for k in range(0,M):
    if (k==0):

        net_output = step_func(mynet,input_test_torch[0,:], time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    else:
        net_output = step_func(mynet,torch.from_numpy(net_pred[k-1,:]).float().cuda(), time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    if k%10000==0:
        print(k)        

print('Eval Finished')

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
truth_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)
net_pred_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)

for n in range(np.shape(net_pred)[0]):
    truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    net_pred_fspec_x[n,:] = np.abs(np.fft.fft(net_pred[n,:])) 


# calculate time derivative using 1st order finite diff
truth_dt = np.diff(label_test, n=1, axis=0)
net_pred_dt = np.diff(net_pred, n=1, axis=0)

# calculate fourier spectrum of time derivative along a single timestep
truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
net_pred_fspec_dt = np.zeros(np.shape(net_pred_dt[:,:]), dtype=complex)


for n in range(np.shape(truth_dt)[0]):
    truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    net_pred_fspec_dt[n,:] = np.abs(np.fft.fft(net_pred_dt[n,:])) 



matfiledata_output = {}
matfiledata_output[u'prediction'] = net_pred
matfiledata_output[u'Truth'] = label_test 
matfiledata_output[u'RMSE'] = RMSE(net_pred, label_test)
matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
matfiledata_output[u'pred_FFT_x'] = net_pred_fspec_x
matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
matfiledata_output[u'pred_FFT_dt'] = net_pred_fspec_dt

scipy.io.savemat(path_outputs+eval_output_name+'.mat', matfiledata_output)

temp_matfile = {}
temp_matfile[u'RMSE'] = matfiledata_output[u'RMSE']
scipy.io.savemat(path_outputs+eval_output_name+'_RMSE.mat', temp_matfile)

if skip_factor: #check if not == 0
    matfiledata_output_skip = {}
    matfiledata_output_skip[u'prediction'] = net_pred[0::skip_factor,:]
    matfiledata_output_skip[u'Truth'] = label_test[0::skip_factor,:]
    matfiledata_output_skip[u'RMSE'] = RMSE(net_pred, label_test)[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_x'] = net_pred_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_fspec_dt[0::skip_factor,:]

    scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matfiledata_output_skip)
print('Data saved')