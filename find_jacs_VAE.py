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
from nn_VAE import VAE
# from nn_step_methods import Directstep, Eulerstep, PECstep

from torch.profiler import profile, record_function, ProfilerActivity
import gc

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print('lead ',lead)

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/jacobean_mats/'

model_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/Conrad_Research_4_Ashesh/VAE_PEC4step_lead1.pt"

matfile_name = 'VAE_PEC4step_lead'+str(lead)+'_jacs_V2.mat'


print('loading data')

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

print('data loaded')


trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float()
label_test = np.transpose(data[:,trainN+lead:])



num_ensembles = 10
eq_points = 1
# FNO archetecture hyperparams

x_torch = torch.zeros([eq_points,input_size]).cuda()

# count=0
# for k in (np.array([ int(0),  int(10000), int(20000), int(99999)])):
#   x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
#   count=count+1

count=0
for k in (np.array([int(0)])):
  x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
  count=count+1

imgChannels = 1
out_channels = 1
num_filters = 64
dimx = 1024
dimy = 1
zDim = 256
learning_rate = 0.001
lr_decay = 0.4

mynet = VAE(imgChannels, out_channels, num_filters, dimx, dimy, zDim).cuda()

mynet.load_state_dict(torch.load(model_path))
print('model defined')
print(model_path)

mynet.cuda()




def Eulerstep(input_batch):
 output_1, mu, sigma = mynet(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1)
  
def Directstep(input_batch):
  output_1, mu, sigma = mynet(input_batch.cuda())
  return output_1

def RK4step(input_batch):
 output_1, mu1, sigma1 = mynet(input_batch.cuda())
 output_2, mu2, sigma2 = mynet(input_batch.cuda()+0.5*output_1)
 output_3, mu3, sigma3 = mynet(input_batch.cuda()+0.5*output_2)
 output_4, mu4, sigma4 = mynet(input_batch.cuda()+output_3)
 output_full = input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6
 mu_full = torch.mean(torch.stack([mu1, mu2, mu3, mu4], dim=0), dim=0)
 sigma_full = torch.mean(torch.stack([sigma1, sigma2, sigma3, sigma4], dim=0), dim=0)
 return output_full

def PECstep(input_batch):
 output_1, mu, sigma = mynet(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_1))

def PEC4step(input_batch):
 output_0, mu1, sigma1 = mynet(input_batch.cuda())
 output_1 = time_step*(output_0) + input_batch.cuda()
 output_2, mu2, sigma2 = mynet(output_1)
 output_2 = input_batch.cuda() + time_step*0.5*(output_0+output_2)
 output_3, mu3, sigma3 = mynet(output_2)
 output_3 = input_batch.cuda() + time_step*0.5*(output_0+output_3)
 output_4, mu4, sigma4 = mynet(output_3)
 output_4 = time_step*0.5*(output_0+output_4)
 mu_full = torch.mean(torch.stack([mu1, mu2, mu3, mu4], dim=0), dim=0)
 sigma_full = torch.mean(torch.stack([sigma1, sigma2, sigma3, sigma4], dim=0), dim=0)
 return input_batch.cuda() + output_4

# print(torch.cuda.memory_allocated())
step_func = PEC4step

print("step function is "+str(step_func))
# print(torch.cuda.memory_allocated())

ygrad = np.zeros([num_ensembles,eq_points,input_size,input_size])

mynet.eval()


for i in range(num_ensembles):
    for k in range(0,eq_points):
        temp_mat = torch.autograd.functional.jacobian(step_func, x_torch[k,:]).detach().cpu().numpy()
        ygrad [i,k,:,:] = temp_mat
        
        print(sum(sum(np.abs(ygrad[i,k,:,:]))))



print(ygrad.shape)



matfiledata = {}
matfiledata[u'Jacobian_mats'] = ygrad
scipy.io.savemat(path_outputs+matfile_name, matfiledata)

print('Saved Predictions')

