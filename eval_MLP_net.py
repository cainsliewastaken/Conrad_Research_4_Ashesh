import numpy as np
import scipy.io
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
#import netCDF4 as nc
#from prettytable import PrettyTable
#from count_trainable_params import count_parameters    
import pickle
import matplotlib.pyplot as plt
from nn_MLP import MLP_NET
from nn_step_methods import directstep, Eulerstep, RK4step, PECstep, PEC4step



path_outputs = '/media/volume/sdb/conrad_stability/model_eval/' #this is where the saved graphs and .mat files end up

net_file_name = 'NN_directstep_lead1.pt' #change this to use a different network

step_func = directstep #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f: #change for eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN=150000 #dont explicitly need this as no training is done in file
input_size = 1024
hidden_layer_size = 2000
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])

my_net_MLP = MLP_NET(input_size, hidden_layer_size, output_size)

M = np.size(label_test,0)
net_pred = np.zeros([M,np.size(label_test,1)])



for k in range(0,M):
 
    if (k==0):

        net_output = step_func(my_net_MLP,input_test_torch[0,:], time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    else:

        net_output = step_func(my_net_MLP,torch.from_numpy(net_pred[k-1,:]).float().cuda(), time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

       

