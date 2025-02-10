import torch
import torch.nn as nn
import torch.optim as optim

class EnsKFstep_module(nn.Module):
    def __init__(self, observation_func, ydim):
        super(EnsKFstep_module, self).__init__()
        self.observation_func = observation_func
        self.y_true_mean = nn.Parameter(torch.randn(1,ydim))
        # self.y_true_mean = torch.zeros((1, ydim), requires_grad = False).cuda()
        
        self.y_true_std = nn.Parameter(torch.randn(1,ydim))
        # self.y_true_std = torch.ones((1, ydim), requires_grad = False).cuda()

        # self.scale_fac = nn.Parameter(torch.randn(1,1))

    def forward(self, Ens_ten):
        # Ens_ten is the tensor of shape batch x ens x input_size
        batch_size = Ens_ten.shape[0]
        ens_size = Ens_ten.shape[1]

        Y = self.observation_func(Ens_ten)  # shape batch x ens x ydim
        
        x_m = torch.mean(Ens_ten, dim=1, keepdim=True)  # shape batch x 1 x input_size
        y_m = torch.mean(Y, dim=1, keepdim=True)  # shape batch x 1 x ydim
        
        # Subtract the means
        X = Ens_ten - x_m  # shape batch x ens x input_size
        Y = Y - y_m  # shape batch x ens x ydim

        # Transpose and reshape for matrix multiplication (swap batch and ens dim for mm)
        X = X.transpose(1, 2)  # shape batch x input_size x ens
        Y = Y.transpose(1, 2)  # shape batch x ydim x ens
        
        # Covariance matrices: perform batch-wise matmuls
        C_xy = torch.bmm(X, Y.transpose(1, 2)) / ens_size  # shape batch x input_size x ydim
        C_yy = torch.bmm(Y, Y.transpose(1, 2)) / ens_size  # shape batch x ydim x ydim
        # print(X.shape, Y.shape, C_xy.shape, C_yy.shape)

        # Add noise to diagonal (R matrix), batch-wise
        #R = torch.diag_embed(torch.randn(batch_size, Y.shape[1]).cuda() * self.y_true_std)  # shape batch x ydim x ydim
        R = torch.diag_embed(torch.square(self.y_true_std)) # (y by y) * y = y
        
        # Kalman gain, batch-wise
        # K = torch.bmm(C_xy, torch.inverse(R + C_yy))  # shape batch x input_size x ydim
        K=torch.linalg.solve(R+C_yy,C_xy,left = False)
        # Update the ensemble

        R_2 = torch.randn(Y.shape).cuda() * torch.square(self.y_true_std).unsqueeze(2) #add R_2 to add noise to invarient measure
        update_term = torch.bmm(K, (self.y_true_mean.unsqueeze(0).transpose(1, 2) - Y + R_2))  # shape batch x input_size x ens
        Ens_ten_out = Ens_ten + update_term.transpose(1, 2)  # shape batch x ens x input_size
        
        return Ens_ten_out
 