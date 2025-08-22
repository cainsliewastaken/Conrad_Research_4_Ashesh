import numpy as np
import scipy.io
import torch
import os
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

skip_factor = 0 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead,'lead')

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/Net_output_pred_jacs/' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_lead1_jacobian_multistep/chkpt_FNO_Eulerstep_lead1_jacobian_multistep_epoch_60.pt"
print(net_file_name)
#change this to use a different network

step_func = Euler_step #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

eval_output_name = 'KS_pred_Eulerstep_FNO_jacs_multistep_many_init_conds'  # what to name the output file, .mat ending not needed
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

# num_iters = 10
# step_method = step_func(my_net, device, num_iters, time_step)  #for implicit methods

# M = int(np.floor(99998/lead))
M = label_test_torch.shape[0] - 1
M = 100
num_init_conds = 10
net_pred = np.zeros([num_init_conds, M, np.size(label_test,1)])
net_pred_truth = np.zeros([num_init_conds, M, np.size(label_test,1)])

print(M)
print('Model loaded')

ygrad = torch.zeros([num_init_conds, M, input_size, input_size])
ygrad_truth = torch.zeros([num_init_conds, M ,input_size, input_size])

label_test_torch = label_test_torch[:label_test_torch.shape[0] - (label_test_torch.shape[0]%num_init_conds)]
label_multi = torch.stack(torch.chunk(label_test_torch, num_init_conds, dim=0), dim=0)
label_multi = label_multi[:, :M]

print(label_multi.shape)
# jac_func_single = torch.func.jacrev(step_method)
# jac_func = torch.func.vmap(jac_func_single, in_dims=())

def batch_jacobian_func(input_tens):
    ygrad = torch.zeros(input_tens.shape[0], input_size, input_size)
    for num in range(0, input_tens.shape[0]):
        ygrad[num] = torch.autograd.functional.jacobian(step_method, torch.reshape(input_tens[int(num)],(1, input_size,1))).reshape(1,1024,1024)
    return ygrad

for k in range(0,M):
 
    if (k==0):

        net_output = step_method(torch.reshape(label_multi[:,0,:].cuda(),(num_init_conds, input_size,1)))
        net_pred [:, k,:] = torch.reshape(net_output,(num_init_conds, input_size)).detach().cpu().numpy()
        net_output_truth = step_method(torch.reshape(label_multi[:,0,:].cuda(),(num_init_conds, input_size,1)))
        net_pred_truth[:, k,:] = torch.reshape(net_output_truth,(num_init_conds, input_size)).detach().cpu().numpy()
        # print(sum(sum(abs(net_pred))))
        # temp_mat = torch.autograd.functional.jacobian(step_method, torch.reshape(label_multi[:,k,:],(num_init_conds, input_size,1))) #Use these for FNO
        ygrad[:, k] = batch_jacobian_func(label_multi[:,0,:])
        ygrad_truth[:,k] = batch_jacobian_func(label_multi[:,0,:])
        # ygrad [:, k] = jac_func(torch.reshape(label_multi[:,0,:],(num_init_conds, input_size,1))).reshape(num_init_conds, input_size, input_size)
        # ygrad_truth[:,k] = jac_func(torch.reshape(label_multi[:,0,:],(num_init_conds, input_size,1))).reshape(num_init_conds, input_size, input_size)

    else:

        net_output = step_method(torch.reshape(torch.from_numpy(net_pred[:, k-1,:]),(num_init_conds, input_size,1)).float().cuda())
        net_pred [:, k,:] = torch.reshape(net_output,(num_init_conds, input_size)).detach().cpu().numpy()

        net_output_truth = step_method(torch.reshape(label_multi[:, k,:].cuda(),(num_init_conds, input_size,1)))
        net_pred_truth[:, k,:] = torch.reshape(net_output_truth,(num_init_conds, input_size)).detach().cpu().numpy()

        # ygrad [:, k] = jac_func(torch.reshape(net_pred[:, k-1,:],(num_init_conds, input_size,1))).reshape(num_init_conds, input_size, input_size)
        # ygrad_truth[:,k] = jac_func(torch.reshape(label_multi[:, k, :],(num_init_conds, input_size,1))).reshape(num_init_conds, input_size, input_size)
        ygrad[:, k] = batch_jacobian_func(torch.tensor(net_pred[:, k-1,:]).float())
        ygrad_truth[:,k] = batch_jacobian_func(label_multi[:, k, :])


    if k%1==0:
        print(k) 
       
print('Eval Finished')

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

# #this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
# truth_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)
# net_pred_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)

# for n in range(np.shape(net_pred)[0]):
#     truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
#     net_pred_fspec_x[n,:] = np.abs(np.fft.fft(net_pred[n,:])) 

# # calculate time derivative using 1st order finite diff
# truth_dt = np.diff(label_test, n=1, axis=0)
# net_pred_dt = np.diff(net_pred, n=1, axis=0)

# # calculate fourier spectrum of time derivative along a single timestep
# truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
# net_pred_fspec_dt = np.zeros(np.shape(net_pred_dt[:,:]), dtype=complex)

# for n in range(np.shape(net_pred_dt)[0]):
#     truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
#     net_pred_fspec_dt[n,:] = np.abs(np.fft.fft(net_pred_dt[n,:])) 


ygrad = ygrad.detach().cpu().numpy()

def RMSE(y_hat, y_true):
    return torch.sqrt(torch.mean((y_hat - y_true)**2, axis=2, keepdims=True)) 

def calc_save_chunk(net_pred_chunk, net_pred_truth_chunk, label_test_chunk, chunk_num, ygrad_chunk, ygrad_truth_chunk):
    # pred_RMSE = np.zeros([num_init_conds, net_pred_chunk.shape[0]])
    # truth_fspec_x = np.zeros(np.shape(net_pred_chunk[:,:]), dtype=complex)
    # net_pred_chunk_fspec_x = np.zeros(np.shape(net_pred_chunk[:,:]), dtype=complex)
    # truth_dt = np.diff(label_test_chunk, n=1, axis=0)
    # net_pred_chunk_dt = np.diff(net_pred_chunk, n=1, axis=0)
    # truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
    # net_pred_chunk_fspec_dt = np.zeros(np.shape(net_pred_chunk_dt[:,:,:]), dtype=complex)

    #this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes

    # pred_RMSE = RMSE(torch.tensor(net_pred_chunk), torch.tensor(label_test_chunk[:, 0:net_pred_chunk.shape[0]])).reshape(num_init_conds, net_pred_chunk.shape[1])
    # pred_truth_RMSE = RMSE(net_pred_truth_chunk, label_test_chunk[:, 0:net_pred_chunk.shape[0]]).reshape(num_init_conds, net_pred_chunk.shape[1])

    # for n in range(np.shape(net_pred_chunk)[0]):
    #     # truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test_chunk[n,:])) 
    #     net_pred_chunk_fspec_x[n ,:] = np.abs(np.fft.fft(net_pred_chunk[n, :])) 

    # for n in range(np.shape(net_pred_chunk_dt)[0]):
    #     # truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    #     net_pred_chunk_fspec_dt[n, ens, :] = np.abs(np.fft.fft(net_pred_chunk_dt[n, ens, :])) 

    # net_pred_chunk_fspec_x = np.abs(np.fft.fft(net_pred_chunk[:], axis=1)) 
    print('Calculation Finished')

    matfiledata_output = {}
    matfiledata_output[u'prediction'] = net_pred_chunk
    matfiledata_output[u'prediction_truth'] = net_pred_truth_chunk
    matfiledata_output[u'Truth'] = label_test_chunk
    # matfiledata_output[u'RMSE'] = pred_RMSE
    # matfiledata_output[u'RMSE_truth'] = pred_truth_RMSE
    # matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
    # matfiledata_output[u'pred_FFT_x'] = net_pred_chunk_fspec_x
    # matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
    # matfiledata_output[u'pred_FFT_dt'] = net_pred_chunk_fspec_dt
    matfiledata_output[u'Jacobians'] = ygrad_chunk
    matfiledata_output[u'Jacobians_truth'] = ygrad_truth_chunk


    print('First save done')
    np.save(path_outputs+'/'+eval_output_name+'/'+eval_output_name+'_chunk_'+str(chunk_num), matfiledata_output)


    print('Saved main file')

    if skip_factor!=0: #check if not == 0
        matfiledata_output_skip = {}
        matfiledata_output_skip[u'prediction'] = net_pred_chunk[:, 0::skip_factor,:]
        matfiledata_output_skip[u'prediction_truth'] = net_pred_truth_chunk[:, 0::skip_factor,:]
        matfiledata_output_skip[u'Truth'] = label_test_chunk[:, 0::skip_factor,:]
        # matfiledata_output_skip[u'RMSE'] = pred_RMSE[:, 0::skip_factor,:]
        # matfiledata_output_skip[u'RMSE_truth'] = pred_truth_RMSE[:, 0::skip_factor,:]
        # matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[:, 0::skip_factor,:]
        # matfiledata_output_skip[u'pred_FFT_x'] = net_pred_chunk_fspec_x[:, 0::skip_factor,:]
        # matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[:, 0::skip_factor,:]
        # matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_chunk_fspec_dt[:, 0::skip_factor,:,:]
        matfiledata_output_skip[u'Jacobians'] = ygrad_chunk[:, 0::skip_factor,:]
        matfiledata_output_skip[u'Jacobians_truth'] = ygrad_truth_chunk[:, 0::skip_factor,:]

        
        np.save(path_outputs+'/'+eval_output_name+'/'+eval_output_name+'_skip'+str(skip_factor)+'_chunk_'+str(chunk_num), matfiledata_output_skip)



if not os.path.exists(path_outputs+'/'+eval_output_name+'/'):
    os.makedirs(path_outputs+'/'+eval_output_name+'/')
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' created.")
else:
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' already exists.")

prev_ind = 0
chunk_count = 0
num_chunks = 100
for chunk in np.array_split(net_pred, num_chunks, axis=1):
    current_ind = prev_ind + chunk.shape[1]
    calc_save_chunk(chunk, net_pred_truth[:, prev_ind:current_ind], label_multi[:, 1+prev_ind:1+current_ind],  chunk_count, ygrad[:, prev_ind:current_ind], ygrad_truth[:, prev_ind:current_ind])
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')
