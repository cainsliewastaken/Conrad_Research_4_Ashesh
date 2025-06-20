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
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.gelu, use_act = True, use_dropout=True):
        super().__init__()
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_in.weight)
        self.activation = activation
        self.layers_1 = nn.ModuleList()
        if use_dropout:
            self.drop = nn.ModuleList()
        self.use_drop = use_dropout
        self.num_layers = num_layers
        self.use_act = use_act
        for i in range(0,num_layers): 
            self.layers_1.append(nn.Linear(hidden_dim, hidden_dim))
            torch.nn.init.xavier_normal_(self.layers_1[i].weight)
            if use_dropout:
                self.drop.append(nn.Dropout(p=0.9))
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        torch.nn.init.xavier_normal_(self.linear_out.weight)

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        x_0 = x
        for i in range(0,self.num_layers):
            x = self.activation(self.layers_1[i](x)) 
            if self.use_drop:
                x = self.drop[i](x)
            # x = x + x_0
        x = self.linear_out(x)
        if self.use_act:
            x = self.activation(x)
        return x
    
    
    def to_spherical(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Code taken from: https://github.com/mwoedlinger/n-sphere-pytorch/blob/main/transformations.py

        Convert Cartesian coordinates to n-dimensional spherical coordinates.

        Args:
            latent_state (torch.Tensor): Tensor representing Cartesian coordinates (x_1, ... x_n).
                                Shape: (..., n)

        Returns:
            torch.Tensor: Tensor representing spherical coordinates (r, phi_1, ... phi_n-1).
                        Shape: (..., n)
        """    
        eps=1e-7
        n = latent_state.shape[-1]
        
        # We compute the coordinates following https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        r = torch.norm(latent_state, dim=-1, keepdim=True)

        # phi_norms are the quotients in the wikipedia article above
        phi_norms = torch.norm(torch.tril(latent_state.flip(-1).unsqueeze(-2).expand((*latent_state.shape, n))), dim=-1).flip(-1)
        phi = torch.arccos(torch.clamp(latent_state[..., :-2]/(phi_norms[..., :-2]), -1 + eps, 1 - eps ))
        phi_final = torch.arccos(torch.clamp(latent_state[..., -2:-1]/(phi_norms[..., -2:-1]), -1 + eps, 1 - eps )) + (2*torch.pi - 2*torch.arccos(torch.clamp(latent_state[..., -2:-1]/(phi_norms[..., -2:-1]+eps), -1 + eps, 1 - eps )))*(latent_state[..., -1:] < 0)
   
        return torch.cat([r, phi, phi_final], dim=-1)
    
    
    def KDE_uniform_loss(self, latent_state, alpha, beta, bandwidth=None):
        # spher_latent_state B x l
        spher_latent_state = self.to_spherical(latent_state)
        r_vals = spher_latent_state[:,0]
        n_bins = int(spher_latent_state.shape[0]/4)
        spher_latent_state = spher_latent_state[:,1:]
        data_min = -torch.pi
        data_max = torch.pi
        bin_edges = torch.linspace(data_min, data_max, n_bins + 1).to(spher_latent_state.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) # k 
        
        if bandwidth is None:
            bandwidth = (data_max - data_min) / n_bins

        histogram = torch.exp(-0.5 * ((bin_centers.reshape([-1, 1, 1]) - spher_latent_state.unsqueeze(0)) / bandwidth)**2).sum(1) #k x L
        histogram = histogram / histogram.sum()

        uniform_vec = torch.ones(n_bins, requires_grad=False).cuda()
        uniform_vec = uniform_vec / uniform_vec.sum()

        return alpha*torch.mean((r_vals - 1)**2) + beta*((uniform_vec.unsqueeze(-1) - histogram)**2).sum(0).mean()