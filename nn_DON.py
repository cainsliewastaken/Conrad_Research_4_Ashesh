import torch
import deepxde.nn.pytorch as deepxde_torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet_FullGrid(nn.Module):
    def __init__(self, layer_sizes_branch: list, layer_sizes_trunk: list, input_size: int):
        super(DeepONet_FullGrid, self).__init__()
        self.input_size = input_size
        self.DeepONet1 = deepxde_torch.deeponet.DeepONet(layer_sizes_branch, layer_sizes_trunk, 'tanh', "Glorot uniform", 1, None).cuda()
        self.DeepONet2 = deepxde_torch.deeponet.DeepONet(layer_sizes_branch, layer_sizes_trunk, 'tanh', "Glorot uniform", 1, None).cuda()
        self.DeepONet3 = deepxde_torch.deeponet.DeepONet(layer_sizes_branch, layer_sizes_trunk, 'tanh', "Glorot uniform", 1, None).cuda()
        self.input_grid = torch.linspace(-1,1, input_size)
        self.tanh = nn.Tanh()


    def forward(self,x):
        x1 = self.DeepONet1((x.unsqueeze(0), self.input_grid.unsqueeze(-1)))
        x1 = self.tanh(x1.squeeze(-1))

        x2 = self.DeepONet1((x1.unsqueeze(0), self.input_grid.unsqueeze(-1)))
        x2 = self.tanh(x2.squeeze(-1))

        x3 = self.DeepONet1((x2.unsqueeze(0), self.input_grid.unsqueeze(-1)))

        return x3.squeeze(-1)