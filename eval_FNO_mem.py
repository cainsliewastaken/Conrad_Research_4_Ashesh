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
from nn_step_methods import directstep, Eulerstep, PECstep

from torch.profiler import profile, record_function, ProfilerActivity



path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])



lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


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




# FNO models and predictions
mynet_directstep_FNO = FNO1d(modes, width, time_future, time_history).float()
mynet_directstep_FNO.load_state_dict(torch.load('NN_FNO_Directstep_lead1.pt'))
mynet_directstep_FNO.cuda()

mynet_Eulerstep_FNO = FNO1d(modes, width, time_future, time_history).float()
mynet_Eulerstep_FNO.load_state_dict(torch.load('NN_FNO_Eulerstep_lead1.pt'))
mynet_Eulerstep_FNO.cuda()

mynet_PECstep_FNO = FNO1d(modes, width, time_future, time_history).float()
mynet_PECstep_FNO.load_state_dict(torch.load('NN_FNO_PECstep_lead1.pt'))
mynet_PECstep_FNO.cuda()

val_dict_direct_FNO = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_directstep_1024_FNO_lead1.mat')
val_dict_Euler_FNO = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_Eulerstep_1024_FNO_lead1.mat')
val_dict_PEC_FNO = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_PECstep_1024_FNO_lead1.mat')

pred_direct_FNO = val_dict_direct_FNO[u'prediction']
pred_Euler_FNO = val_dict_Euler_FNO[u'prediction']
pred_PEC_FNO = val_dict_PEC_FNO[u'prediction']

ygrad_direct_FNO = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_FNO = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_FNO = torch.zeros([int(4),input_size,input_size])


with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
    PECstep(mynet_PECstep_FNO,torch.reshape(input_test_torch[0,:],(1,input_size,1)))

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))





# i = 0
# for j in np.array([0, 10000, 50000, 99998]):
#     # FNO jacobian calc

#     ygrad_direct_FNO[i,:,:] = torch.func.jacfwd(directstep, argnums=1)(mynet_directstep_FNO, torch.reshape(torch.tensor(pred_direct_FNO[0,:], dtype=torch.float),(1,input_size,1)))
#     ygrad_Euler_FNO[i,:,:] = torch.func.jacfwd(Eulerstep, argnums=1)(mynet_Eulerstep_FNO, torch.reshape(torch.tensor(pred_Euler_FNO[0,:], dtype=torch.float),(1,input_size,1)))
#     ygrad_PEC_FNO[i,:,:] = torch.func.jacfwd(PECstep, argnums=1)(mynet_PECstep_FNO, torch.reshape(torch.tensor(pred_PEC_FNO[0,:], dtype=torch.float),(1,input_size,1)))
#     i += 1


# print('FNO basic jacs calculated')

# del mynet_directstep_FNO, mynet_Eulerstep_FNO, mynet_PECstep_FNO
# torch.cuda.empty_cache()

