import numpy as np
import scipy.io
import torch
import math
import torch.nn as nn

def ensemble_FP(Ens_ten, observation_func, y_true_mean, y_true_std, alpha, beta):
    # Ens_mat is the tensor of ensembles, with ens x (spacial)
    Y = observation_func(Ens_ten) # shape ens by y

    x_m = torch.mean(Ens_ten, dim=0)
    y_m = torch.mean(Y, dim=0)
    X = torch.transpose(Ens_ten - x_m, 0, 1).squeeze(2) # shape x by ens
    Y = torch.transpose(Y - y_m, 0, 1).squeeze(2)# shape y by ens
    
    # if (torch.linalg.vector_norm(X, dim=0).max() - torch.linalg.vector_norm(X, dim=0).min()).abs() < 1e-4:
    #     print('Did not run')
    #     return Ens_ten
    
    C_xy = beta*(torch.mm(X, torch.transpose(Y, 0, 1)))/(Ens_ten.shape[0]-1) # shape x by y
    C_yy = beta*(Y @ Y.transpose(0,1))/(Ens_ten.shape[0]-1) # shape y by y

    # R = torch.diag(torch.randn(Y.shape[0]).cuda())*torch.square(y_true_std) # (y by y) * y = y

    R = torch.diag(torch.ones(Y.shape[0]).cuda())*torch.square(y_true_std) # (y by y) * y = y

    # K = torch.mm(C_xy, torch.inverse(R + C_yy))  # X @ A^-1 = B

    K = torch.linalg.solve(R + C_yy, C_xy, left=False)
        # shape (x by y) * [(y by y) + (y by y)]


    # print('Input max and min ',Ens_ten.max().cpu().numpy(), Ens_ten.min().cpu().numpy() ,'X norm max and min ', torch.linalg.vector_norm(X, dim=0).max().cpu().numpy(), torch.linalg.vector_norm(X, dim=0).min().cpu().numpy())
    # print('Y max and min ', torch.linalg.vector_norm(Y, dim=0).max().cpu().numpy(), torch.linalg.vector_norm(Y, dim=0).min().cpu().numpy(),'Y cov max and min ', C_yy.max().cpu().numpy(), C_yy.min().cpu().numpy())
    # print('Cond number of (R + C_yy) and K', np.linalg.cond((R + C_yy).cpu().numpy()), np.linalg.cond((K).cpu().numpy()))
    # print('')

    R_2 = torch.randn_like(Y).transpose(0,1).cuda()*torch.square(y_true_std).expand(Ens_ten.shape[0],-1) #y by ens * y by 1
    Ens_ten_out = Ens_ten.squeeze(2) + torch.transpose(torch.mm(K, (torch.transpose(y_true_mean,0,1) - Y.mean(1).unsqueeze(1) + alpha*R_2.transpose(0,1))), 0, 1) # ens by x + (x by y)*(1 by y - y by ens + y by ens)

    return Ens_ten_out.unsqueeze(2)

def ensemble_FP_cov(Ens_ten, observation_func, y_true_mean, y_true_cov, alpha, beta):
    # Ens_mat is the tensor of ensembles, with ens x (spacial)
    Y = observation_func(Ens_ten) # shape ens by y

    x_m = torch.mean(Ens_ten, dim=0)
    y_m = torch.mean(Y, dim=0)
    X = torch.transpose(Ens_ten - x_m, 0, 1).squeeze(2) # shape x by ens
    Y = torch.transpose(Y - y_m, 0, 1).squeeze(2)# shape y by ens
    
    # if (torch.linalg.vector_norm(X, dim=0).max() - torch.linalg.vector_norm(X, dim=0).min()).abs() < 1e-4:
    #     print('Did not run')
    #     return Ens_ten

    C_xy = beta*(torch.mm(X, torch.transpose(Y, 0, 1)))/(Ens_ten.shape[0]-1) # shape x by y
    C_yy = beta*(Y @ Y.transpose(0,1))/(Ens_ten.shape[0]-1) # shape y by y

    # R = torch.diag(torch.randn(Y.shape[0]).cuda())*torch.square(y_true_std) # (y by y) * y = y

    # R = torch.diag(torch.ones(Y.shape[0]).cuda())*torch.square(y_true_std) # (y by y) * y = y

    K = torch.mm(C_xy, torch.inverse(y_true_cov + C_yy))
        # shape (x by y) * [(y by y) + (y by y)]

    # print('Input max and min ',Ens_ten.max().cpu().numpy(), Ens_ten.min().cpu().numpy() ,'X norm max and min ', torch.linalg.vector_norm(X, dim=0).max().cpu().numpy(), torch.linalg.vector_norm(X, dim=0).min().cpu().numpy())
    # print('Y max and min ', torch.linalg.vector_norm(Y, dim=0).max().cpu().numpy(), torch.linalg.vector_norm(Y, dim=0).min().cpu().numpy(),'Y cov max and min ', C_yy.max().cpu().numpy(), C_yy.min().cpu().numpy())
    # print('Cond number of R, C_yy, (R + C_yy) and C_xy', np.linalg.cond((y_true_cov).cpu().numpy()), np.linalg.cond(C_yy.cpu().numpy()), np.linalg.cond((y_true_cov+C_yy).cpu().numpy()), np.linalg.cond(C_xy.cpu().numpy()))
    # print('')
    
    # Ens_ten_out = Ens_ten.squeeze(2) + torch.transpose(torch.mm(K, (torch.transpose(y_true_mean,0,1) - Y)), 0, 1)

    R_2 = y_true_cov @ torch.randn_like(Y).cuda() #y by y * y by ens 
    Ens_ten_out = Ens_ten.squeeze(2) + torch.transpose(torch.mm(K, (torch.transpose(y_true_mean,0,1) - Y + alpha*R_2)), 0, 1) # ens by x + (x by y)*(1 by y - y by ens + y by ens)

    return Ens_ten_out.unsqueeze(2)
