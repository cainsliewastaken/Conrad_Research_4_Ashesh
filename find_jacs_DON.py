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
from count_trainable_params import count_parameters    
#import hdf5storage
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from nn_DON import DeepONet_FullGrid 
from nn_MLP import MLP_Net 
from nn_step_methods import Directstep, Eulerstep, PEC4step, RK4step

import time



time_step = 1e-3
lead = 1
print(lead)

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/jacobean_mats/'

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/DON_PEC4step_lead1/chkpt_DON_PEC4step_lead1_epoch100.pt"
# net_file_name = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/MLP_PEC4step_lead1/chkpt_MLP_PEC4step_lead1_epoch60.pt'
print(net_file_name)
matfile_name = 'DON_PEC4step_lead'+str(lead)+'_jacs.mat'


print('loading data')

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

print('data loaded')


trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).double()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).double()
label_test = np.transpose(data[:,trainN+lead:])


eq_points = 4


x_torch = torch.zeros([eq_points,input_size]).cuda()

count=0
for k in (np.array([ int(0),  int(10000), int(20000), int(99999)])):
  x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
  count=count+1


# mynet = MLP_Net(input_size, hidden_layer_size, output_size)



num_of_basis_funcs = 200
layer_sizes_branch = [1024, 4096, 4096, 4096, num_of_basis_funcs]
layer_sizes_trunk = [1, 521, 512, 512, num_of_basis_funcs]

mynet = DeepONet_FullGrid(layer_sizes_branch, layer_sizes_trunk, 1024)
mynet.load_state_dict(torch.load(net_file_name))
mynet.cuda()


print('model defined')
print(net_file_name)



def RK4step(input_batch):
 output_1 = mynet(input_batch.cuda())
 output_2 = mynet(input_batch.cuda()+0.5*output_1)
 output_3 = mynet(input_batch.cuda()+0.5*output_2)
 output_4 = mynet(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6

def Eulerstep(input_batch):
 output_1 = mynet(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 
  
def Directstep(input_batch):
  output_1 = mynet(input_batch.cuda())
  return output_1

def PECstep(input_batch):
 output_1 = mynet(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_1))


def PEC4step(input_batch):
 output_1 = time_step*mynet(input_batch.cuda()) + input_batch.cuda()
 output_2 = input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_1))
 output_3 = input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_2))
 return input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_3))



step_func = Directstep

print("step function is "+str(step_func))
count_parameters(mynet)


ygrad = torch.zeros([eq_points,input_size,input_size])

mynet.eval()
mynet.double()

t_0 = time.time()
for k in range(0,eq_points):
    ygrad [k,:,:] = torch.autograd.functional.jacobian(step_func,x_torch[k,:].double(), strict = True) 
    # ygrad [k,:,:] = torch.func.jacfwd(step_func, argnums=1)(mynet, x_torch[k,:].double(), time_step) 
t_1 = time.time()
print(t_1-t_0, 'Time taken to run')
print(sum(sum(ygrad[1,:,:])))


ygrad = ygrad.detach().cpu().numpy()

print(ygrad.shape)

matfiledata = {}
matfiledata[u'Jacobian_mats'] = ygrad
scipy.io.savemat(path_outputs+matfile_name, matfiledata)

print('Saved Predictions')
