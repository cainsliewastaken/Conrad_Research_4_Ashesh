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
import pickle
import matplotlib.pyplot as plt
from nn_DON import DeepONet_FullGrid
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step



skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead,'lead')

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/DON_output/' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/DON_PEC4step_lead1/chkpt_DON_PEC4step_lead1_epoch100.pt"
#change this to use a different network

step_func = PEC4step #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step
print(step_func)

eval_output_name = 'predicted_PEC4step_1024_DON_lead'+str(lead)+''  # what to name the output file, .mat ending not needed

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f: #change based on eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])



trainN=150000 #dont explicitly need this as no training is done in file
input_size = 1024
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])

num_of_basis_funcs = 200
layer_sizes_branch = [1024, 4096, 4096, 4096, num_of_basis_funcs]
layer_sizes_trunk = [1, 521, 512, 512, num_of_basis_funcs]

my_net_DON = DeepONet_FullGrid(layer_sizes_branch, layer_sizes_trunk, 1024)
count_parameters(my_net_DON)
my_net_DON.load_state_dict(torch.load(net_file_name))
my_net_DON.cuda()


M = int(np.floor(99999/lead))
net_pred = np.zeros([M,np.size(label_test,1)])

print('Model loaded')


for k in range(0,M):
    if (k==0):

        net_output = step_func(my_net_DON,label_test_torch[0,:], time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    else:
        net_output = step_func(my_net_DON,torch.from_numpy(net_pred[k-1,:]).float().cuda(), time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    if k%10000==0:
        print(k)        

print('Eval Finished')


def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
truth_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)
net_pred_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)

for n in range(np.shape(net_pred)[0]):
    truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    net_pred_fspec_x[n,:] = np.abs(np.fft.fft(net_pred[n,:])) 


# calculate time derivative using 1st order finite diff
truth_dt = np.diff(label_test, n=1, axis=0)
net_pred_dt = np.diff(net_pred, n=1, axis=0)

# calculate fourier spectrum of time derivative along a single timestep
truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
net_pred_fspec_dt = np.zeros(np.shape(net_pred_dt[:,:]), dtype=complex)


for n in range(np.shape(truth_dt)[0]):
    truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    net_pred_fspec_dt[n,:] = np.abs(np.fft.fft(net_pred_dt[n,:])) 



matfiledata_output = {}
matfiledata_output[u'prediction'] = net_pred
matfiledata_output[u'Truth'] = label_test 
matfiledata_output[u'RMSE'] = RMSE(net_pred, label_test)
matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
matfiledata_output[u'pred_FFT_x'] = net_pred_fspec_x
matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
matfiledata_output[u'pred_FFT_dt'] = net_pred_fspec_dt

scipy.io.savemat(path_outputs+eval_output_name+'.mat', matfiledata_output)

temp_matfile = {}
temp_matfile[u'RMSE'] = matfiledata_output[u'RMSE']
scipy.io.savemat(path_outputs+eval_output_name+'_RMSE.mat', temp_matfile)


if skip_factor: #check if not == 0
    matfiledata_output_skip = {}
    matfiledata_output_skip[u'prediction'] = net_pred[0::skip_factor,:]
    matfiledata_output_skip[u'Truth'] = label_test[0::skip_factor,:]
    matfiledata_output_skip[u'RMSE'] = RMSE(net_pred, label_test)[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_x'] = net_pred_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_fspec_dt[0::skip_factor,:]

    scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matfiledata_output_skip)
print('Data saved')