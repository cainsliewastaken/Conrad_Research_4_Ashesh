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
# from nn_step_methods import Directstep, Eulerstep, PECstep

from torch.profiler import profile, record_function, ProfilerActivity
import gc


lead = 1
path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/jacobean_mats/'

model_path = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/MLP_PEC4step_lead1_tendency/chkpt_MLP_PEC4step_lead1_tendency'

matfile_name = 'MLP_PEC4step_lead'+str(lead)+'_tendency_jacs_all_chkpts.mat'


print('loading data')

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

print('data loaded')


lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float()
label_test = np.transpose(data[:,trainN+lead:])


eq_points = 1
# FNO archetecture hyperparams

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 512 # input and output chasnnels to the FNO layer




x_torch = torch.zeros([eq_points,input_size]).cuda()

count=0
for k in (np.array([ int(0)])):
  x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
  count=count+1



mynet = MLP_Net(input_size, hidden_layer_size, output_size)
# mynet = FNO1d(modes, width, time_future, time_history).cuda()


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


step_func = PEC4step

print("step function is "+str(step_func))

matfiledata = {}

for epoch_num in range(0,61):
    mynet.load_state_dict(torch.load(model_path+'_epoch'+str(epoch_num)+'.pt'))
    mynet.cuda()

    mynet.eval()
    ygrad = torch.zeros([eq_points,input_size,input_size])

    for k in range(0,eq_points):

        ygrad [k,:,:] = torch.autograd.functional.jacobian(step_func,x_torch[k,:]) #Use this line for MLP networks
        
        # temp_mat = torch.autograd.functional.jacobian(step_func, torch.reshape(x_torch[k,:],(1,input_size,1))) #Use these for FNO
        # ygrad [k,:,:] = torch.reshape(temp_mat,(1,input_size, input_size))

        # print(sum(sum(np.abs(ygrad[k,:,:]))))



    ygrad = ygrad.detach().cpu().numpy()
    matfiledata[u'Jacobian_mats_epoch_'+str(epoch_num)] = ygrad




scipy.io.savemat(path_outputs+matfile_name, matfiledata)

print('Saved Predictions')

