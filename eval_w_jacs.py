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
from nn_MLP import MLP_Net
from nn_step_methods import *

skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 5e-2
lead = int((1/1e-3)*time_step)
print(lead,'lead')

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/Net_output_pred_jacs/' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_lead50/chkpt_FNO_Eulerstep_lead50_epoch53.pt"
print(net_file_name)
#change this to use a different network

step_func = Euler_step #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

eval_output_name = 'KS_pred_Eulerstep_FNO_jacs_many_timesteps_lead_50'  # what to name the output file, .mat ending not needed
print(eval_output_name)

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f: #change based on eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN=150000 #dont explicitly need this as no training is done in file
input_size = 1024
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])
print(label_test_torch.shape)

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters

input_size = 1024
hidden_layer_size = 2000
output_size = 1024

modes = 256 # number of Fourier modes to multiply
width = 256  # input and output chasnnels to the FNO layer

my_net = FNO1d(modes, width, time_future, time_history)
# my_net = MLP_Net(input_size, hidden_layer_size, output_size)
my_net.load_state_dict(torch.load(net_file_name))
my_net.cuda()

step_method = step_func(my_net, device, time_step)

# num_iters = 20
# step_method = step_func(my_net, device, num_iters, time_step)  #for implicit methods

# M = int(np.floor(99998/lead))
M = label_test_torch.shape[0] - 1
net_pred = np.zeros([M,np.size(label_test,1)])
print(M)
print('Model loaded')

noise_var = 0

print('Noise number: ', noise_var)

noised_input = (noise_var)*torch.randn(1,1024).cuda()
noised_input = label_test_torch[0,:].cuda() + noised_input
print(noised_input.size())

for k in range(0,M):
 
    if (k==0):

        net_output = step_method(torch.reshape(noised_input,(1,input_size,1)))
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
        print(sum(sum(abs(net_pred))))

    else:

        net_output = step_method(torch.reshape(torch.from_numpy(net_pred[k-1,:]),(1,input_size,1)).float().cuda())
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()

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

for n in range(np.shape(net_pred_dt)[0]):
    truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    net_pred_fspec_dt[n,:] = np.abs(np.fft.fft(net_pred_dt[n,:])) 

#Finding Jacobians
print('Finding Jacs')
eq_points = [x for x in range(0, int(10000/lead), int(1000/lead))]
print(len(eq_points))

ygrad = torch.zeros([len(eq_points),input_size,input_size])
pred_from_truth = torch.zeros([len(eq_points),3,input_size])
truth_from_truth = torch.zeros([len(eq_points),3,input_size])

for k in range(len(eq_points)):    
    temp_mat = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[eq_points[k],:],(1,input_size,1))) #Use these for FNO
    ygrad [k] = torch.reshape(temp_mat,(1,input_size, input_size))
    truth_from_truth = input_test_torch[eq_points[k]:eq_points[k]+3]
    pred_from_truth[k,0] = input_test_torch[eq_points[k]]
    pred_from_truth[k,1] = step_method(torch.reshape(input_test_torch[eq_points[k]],(1,input_size,1))).reshape(1,input_size)
    pred_from_truth[k,2] = step_method(torch.reshape(pred_from_truth[k,1],(1,input_size,1))).reshape(1,input_size)
    print(k)

ygrad = ygrad.detach().cpu().numpy()
pred_from_truth = pred_from_truth.detach().cpu().numpy()
truth_from_truth = truth_from_truth.detach().cpu().numpy()
truth_indices = eq_points

matfiledata_output = {}
matfiledata_output[u'prediction'] = net_pred
matfiledata_output[u'Truth'] = label_test[1:]
matfiledata_output[u'RMSE'] = RMSE(net_pred, label_test[1:net_pred.shape[0]+1])
matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
matfiledata_output[u'pred_FFT_x'] = net_pred_fspec_x
matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
matfiledata_output[u'pred_FFT_dt'] = net_pred_fspec_dt
matfiledata_output[u'Jacobians'] = ygrad
matfiledata_output[u'Pred_from_Truth'] = pred_from_truth
matfiledata_output[u'Truth_from_Truth'] = truth_from_truth
matfiledata_output[u'Truth_indices'] = truth_indices

# scipy.io.savemat(path_outputs+eval_output_name+'.mat', matfiledata_output)
np.save(path_outputs+eval_output_name, matfiledata_output)

temp_matfile = {}
temp_matfile[u'RMSE'] = matfiledata_output[u'RMSE']
# scipy.io.savemat(path_outputs+eval_output_name+'_RMSE.mat', temp_matfile)
np.save(path_outputs+eval_output_name+'_RMSE', temp_matfile)

if skip_factor: #check if not == 0
    matfiledata_output_skip = {}
    matfiledata_output_skip[u'prediction'] = net_pred[0::skip_factor,:]
    matfiledata_output_skip[u'Truth'] = label_test[1::skip_factor,:]
    matfiledata_output_skip[u'RMSE'] = RMSE(net_pred, label_test[1:net_pred.shape[0]+1])[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_x'] = net_pred_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_fspec_dt[0::skip_factor,:]
    matfiledata_output_skip[u'Jacobians'] = ygrad
    matfiledata_output_skip[u'Pred_from_Truth'] = pred_from_truth
    matfiledata_output_skip[u'Truth_indices'] = truth_indices
    matfiledata_output_skip[u'Truth_from_Truth'] = truth_from_truth

    # scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matfiledata_output_skip)
    np.save(path_outputs+eval_output_name+'_skip'+str(skip_factor), matfiledata_output_skip)

print('Data saved')