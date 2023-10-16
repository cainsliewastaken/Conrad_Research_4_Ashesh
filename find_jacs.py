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
#import hdf5storage
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from nn_FNO import FNO1d
from nn_MLP import MLP_Net 
from nn_step_methods import Directstep, Eulerstep, PECstep

from torch.profiler import profile, record_function, ProfilerActivity


lead = 1
path_outputs = '/media/volume/sdb/conrad_stability/jacobian_mats_all_models/'

model_path = "/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/NN_FNO_Directstep_lead1.pt"

matfile_name = 'FNO_KS_PECstep_tendency_lead1_UNTRAINED_jacs.mat'
#'FNO_KS_Directstep_lead'+str(lead)+'_jacs.mat'

step_func = PECstep

print('loading data')


with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

print('data loaded')


lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


eq_points = 4


# FNO archetecture hyperparams

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



x_torch = torch.zeros([eq_points,input_size])

count=0
for k in (np.array([ int(0),  int(10000), int(20000), int(99999)])):
 x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
 count=count+1



# mynet = MLP_Net(input_size, hidden_layer_size, output_size)
mynet = FNO1d(modes, width, time_future, time_history)
# mynet.load_state_dict(torch.load(model_path))
mynet.cuda()
mynet.eval()

ygrad = torch.zeros([eq_points,input_size,input_size])

for k in range(0,eq_points):

    # ygrad [k,:,:] = torch.autograd.functional.jacobian(step_func,x_torch[k,:]) #Use these 2 lines for MLP networks

    temp_mat = torch.autograd.functional.jacobian(step_func, torch.reshape(torch.tensor(x_torch[k,:]),(1,input_size,1))) #Use these for FNO
    ygrad [k,:,:] = torch.reshape(temp_mat,(1,input_size, input_size))

    # print(sum(sum(ygrad[k,:,:])))



ygrad = ygrad.detach().cpu().numpy()

print(ygrad.shape)



matfiledata = {}
matfiledata[u'Jacobian_mats'] = ygrad
scipy.io.savemat(path_outputs+matfile_name, matfiledata)

print('Saved Predictions')
