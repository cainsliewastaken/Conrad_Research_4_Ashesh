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
from nn_FNO import FNO1d
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
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])



time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 64 # input and output chasnnels to the FNO layer

num_epochs = 1 #set to one so faster computation, in principle 20 is best.  WHERE IS THIS USED, WHAT IT DO?
learning_rate = 0.0001
lr_decay = 0.4
num_workers = 0  #What does this do?

my_net_FNO = FNO1d(modes, width, time_future, time_history).cuda()


M = np.size(label_test,0)
net_pred = np.zeros([M,np.size(label_test,1)])



for k in range(0,M):
 
    if (k==0):


        net_output = step_func(my_net_FNO,torch.reshape(input_test_torch[0,:],(1,input_size,1)))
        my_net_FNO [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()

    else:



        net_output = step_func(my_net_FNO,torch.reshape(torch.from_numpy(net_pred[k-1,:]),(1,input_size,1)).float().cuda())
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
       

