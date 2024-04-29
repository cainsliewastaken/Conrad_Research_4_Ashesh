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
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step



skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead,'lead')

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/FNO_output/' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_RK4step_lead1/chkpt_FNO_RK4step_lead1_tendency_epoch60.pt"
#change this to use a different network

step_func = RK4step #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

eval_output_name = 'predicted_RK4step_1024_FNO_lead'+str(lead)+'_tendence_ensembles'  # what to name the output file, .mat ending not needed

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f: #change based on eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])



trainN=150000 #dont explicitly need this as no training is done in file
input_size = 1024
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])


time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 512  # input and output chasnnels to the FNO layer

num_epochs = 1 #set to one so faster computation, in principle 20 is best.  WHERE IS THIS USED, WHAT IT DO?
learning_rate = 0.0001
lr_decay = 0.4

my_net_FNO = FNO1d(modes, width, time_future, time_history)
my_net_FNO.load_state_dict(torch.load(net_file_name))
my_net_FNO.cuda()


num_ensembles = 25
M = int(np.floor(99999/lead))
net_pred = np.zeros([M,num_ensembles,np.size(label_test,1)])
net_pred_mean_traj = np.zeros([M,np.size(label_test,1)])

print('Model loaded')


for k in range(0,M):
#  print('time step',k)
 if (k==0):

   for ens in range (0,num_ensembles):
    output = step_func(label_test_torch[0,:])
    net_pred[k,ens,:,] = np.squeeze(output)

 else:

   for ens in range (0, num_ensembles):
     output = step_func(net_pred[k-1,ens,:,])
     net_pred[k,ens,:,] = np.squeeze(output)

net_pred = net_pred.detach().cpu().numpy()

print('Eval Finished')


def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

pred_RMSE = np.zeros([M,num_ensembles])
# truth_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)
net_pred_fspec_x = np.zeros(np.shape(net_pred[:,:,:]), dtype=complex)
# truth_dt = np.diff(label_test, n=1, axis=0)
net_pred_dt = np.diff(net_pred, n=1, axis=0)
# truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
net_pred_fspec_dt = np.zeros(np.shape(net_pred_dt[:,:,:]), dtype=complex)

for ens in range(0,num_ensembles):
    #this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes

    pred_RMSE[:,ens] = RMSE(net_pred[:,ens,:], label_test)

    for n in range(np.shape(net_pred)[0]):
        # truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
        net_pred_fspec_x[n, ens ,:] = np.abs(np.fft.fft(net_pred[n, ens, :])) 


    for n in range(np.shape(net_pred_dt)[0]):
        # truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
        net_pred_fspec_dt[n, ens, :] = np.abs(np.fft.fft(net_pred_dt[n, ens, :])) 



matfiledata_output = {}
matfiledata_output[u'prediction'] = net_pred
matfiledata_output[u'Truth'] = label_test 
matfiledata_output[u'RMSE'] = pred_RMSE
# matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
matfiledata_output[u'pred_FFT_x'] = net_pred_fspec_x
# matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
# matfiledata_output[u'pred_FFT_dt'] = net_pred_fspec_dt

# scipy.io.savemat(path_outputs+eval_output_name+'.mat', matfiledata_output)

temp_matfile = {}
temp_matfile[u'RMSE'] = matfiledata_output[u'RMSE']
scipy.io.savemat(path_outputs+eval_output_name+'_RMSE.mat', temp_matfile)


if skip_factor: #check if not == 0
    matfiledata_output_skip = {}
    matfiledata_output_skip[u'prediction'] = net_pred[0::skip_factor,:,:]
    matfiledata_output_skip[u'Truth'] = label_test[0::skip_factor,:]
    matfiledata_output_skip[u'RMSE'] = pred_RMSE[0::skip_factor,:]
    # matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_x'] = net_pred_fspec_x[0::skip_factor,:,:]
    # matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_fspec_dt[0::skip_factor,:,:]

    scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matfiledata_output_skip)
print('Data saved')