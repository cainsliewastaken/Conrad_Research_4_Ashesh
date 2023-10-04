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
# from nn_step_methods import directstep, Eulerstep, PECstep

from torch.profiler import profile, record_function, ProfilerActivity



print('loading data')
path_outputs = '/media/volume/sdb/conrad_stability/model_eval_tendency/'

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
for k in (np.array([ int(1),  int(10000), int(20000), int(99999)])):
 x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
 count=count+1






def RK4step(input_batch):
 output_1 = mynet(input_batch.cuda())
 output_2 = mynet(input_batch.cuda()+0.5*output_1)
 output_3 = mynet(input_batch.cuda()+0.5*output_2)
 output_4 = mynet(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6

def Eulerstep(input_batch):
 output_1 = mynet(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 
  
def directstep(input_batch):
  output_1 = mynet(input_batch.cuda())
  return output_1

def PECstep(input_batch):
 output_1 = mynet(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_1))




# mynet = MLP_Net(input_size, hidden_layer_size, output_size)
mynet = FNO1d(modes, width, time_future, time_history)
mynet.load_state_dict(torch.load("/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/NN_Spectral_Loss_with_tendencyfft_lambda_reg5_directstep_lead1.pt"))
mynet.eval()
mynet.cuda()

ygrad = torch.zeros([eq_points,input_size,input_size])

for k in range(0,eq_points):

    # ygrad [k,:,:] = torch.autograd.functional.jacobian(directstep,x_torch[k,:])

    temp_mat = torch.autograd.functional.jacobian(PECstep, torch.reshape(torch.tensor(x_torch[k,:]),(1,input_size,1)))
    ygrad [k,:,:] = torch.reshape(temp_mat,(1,input_size, input_size))


ygrad = ygrad.detach().cpu().numpy()

print(ygrad.shape)



matfiledata = {}
matfiledata[u'Jacobian_mats'] = ygrad
scipy.io.savemat(path_outputs+'MLP_KS_Directstep_tendency_lead'+str(lead)+'.mat', matfiledata)

print('Saved Predictions')
