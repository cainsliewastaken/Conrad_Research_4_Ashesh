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
from nn_MLP import MLP_Net
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_Cascade_MLP import Cascade_MLP_Net

lead=1
time_step = 1e-3
trainN = 150000 #dont explicitly need this as no training is done in file, here to help separate training data from eval data
input_size = 1024
hidden_layer_size = 2000
num_layers = 6
hidden_layer_size_cascade = 1024
output_size = 1024

skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_tendency/' #this is where the saved graphs and .mat files end up

net_file_name = "/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/NN_Directstep_lead1_tendency.pt" #change this to use a different network

step_func = Directstep #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

eval_output_name = 'predicted_Directstep_1024_lead'+str(lead)+'_tendency'  # what to name the output file, .mat ending not needed

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f: #change for eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


my_net_MLP = MLP_Net(input_size, hidden_layer_size, output_size)
# my_net_MLP = Cascade_MLP_Net(input_size, hidden_layer_size_cascade, output_size, num_layers).cuda()
my_net_MLP.load_state_dict(torch.load(net_file_name))
my_net_MLP.cuda()
print('Model loaded')

M = 99999
net_pred = np.zeros([M,np.size(label_test,1)])


for k in range(0,M):
    if (k==0):

        net_output = step_func(my_net_MLP,input_test_torch[0,:], time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    else:
        net_output = step_func(my_net_MLP,torch.from_numpy(net_pred[k-1,:]).float().cuda(), time_step)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    if k%10000==0:
        print(k)        

print('Eval Finished')

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
truth_fspec_x = np.zeros(np.shape(label_test[:,:]), dtype=complex)
net_pred_fspec_x = np.zeros(np.shape(label_test[:,:]), dtype=complex)

for n in range(np.shape(label_test)[0]):
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