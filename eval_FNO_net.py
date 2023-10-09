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


skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/' #this is where the saved graphs and .mat files end up

net_file_name = "NN_Spectral_Loss_FNO_Directstep_tendency_lambda_reg5lead1.pt" #change this to use a different network

step_func = PECstep #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

lead=1
eval_output_name = 'predicted_PECstep_1024_FNO_tendency_lead'+str(lead)+''  # what to name the output file, .mat ending not needed

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f: #change for eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


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

print('Model loaded')


for k in range(0,M):
 
    if (k==0):


        net_output = step_func(my_net_FNO,torch.reshape(input_test_torch[0,:],(1,input_size,1)), time_step)
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()

    else:



        net_output = step_func(my_net_FNO,torch.reshape(torch.from_numpy(net_pred[k-1,:]),(1,input_size,1)).float().cuda(), time_step)
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
       
print('Eval Finished')


def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
u_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)
pred_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)

for n in range(np.shape(label_test)[0]):
    u_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    pred_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(net_pred[n,:])) 


# calculate time derivative using 2nd order finite diff
u_truth_difft_n2 = np.diff(label_test, n=2, axis=0)
u_pred_diff_t_n2 = np.diff(net_pred, n=2, axis=0)

# calculate fourier spectrum of time derivative along a single timestep
u_truth_difft_n2_fspec = np.zeros(np.shape(u_truth_difft_n2[:,:]), dtype=complex)
u_pred_difft_n2_fspec = np.zeros(np.shape(u_pred_diff_t_n2[:,:]), dtype=complex)


for n in range(np.shape(u_truth_difft_n2)[0]):
    u_truth_difft_n2_fspec[n,:] = np.abs(np.fft.fft(u_truth_difft_n2[n,:])) 
    u_pred_difft_n2_fspec[n,:] = np.abs(np.fft.fft(u_pred_diff_t_n2[n,:])) 




# Saving full .mat file
matfiledata_output = {}
matfiledata_output[u'prediction'] = net_pred
matfiledata_output[u'Truth'] = label_test 
matfiledata_output[u'RMSE'] = RMSE(net_pred, label_test)
matfiledata_output[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_output[u'pred_FFT'] = u_pred_difft_n2_fspec

scipy.io.savemat(path_outputs+eval_output_name+'.mat', matfiledata_output)


# Saving smaller .mat file that skips values
if skip_factor: #check if not == 0
    matfiledata_output_skip = {}
    matfiledata_output_skip[u'prediction'] = net_pred[0::skip_factor,:]
    matfiledata_output_skip[u'Truth'] = label_test[0::skip_factor,:]
    matfiledata_output_skip[u'RMSE'] = RMSE(net_pred, label_test)[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT'] = u_1d_fspec_tdim[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT'] = u_pred_difft_n2_fspec[0::skip_factor,:]

    scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matfiledata_output_skip)

print('Data saved')
