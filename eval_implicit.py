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



time_step = 1e-1
lead = (1/1e-3)*time_step

skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

path_outputs = '/home/exouser/conrad_stability/implicit_outputs/' #this is where the saved graphs and .mat files end up

net_file_name = "/home/exouser/conrad_stability/Conrad_Research_4_Ashesh/MLP_PEC4step_implicit_lead100_tendency.pt" #change this to use a different network

eval_output_name = 'predicted_implicit_PEC4step_1024_lead'+str(lead)+'_spectralloss'  # what to name the output file, .mat ending not needed


with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
hidden_layer_size_cascade = 1024
num_layers = 8
num_iters = 50

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()
du_label_torch = input_train_torch - label_train_torch

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])



def Eulerstep_implicit(net,input_batch,output_iter):
 output_1 = net(output_iter.cuda())
 return input_batch.cuda() + time_step*(output_1)


def PEC4step_implicit(net,input_batch,output_iter):
 output_1 = time_step*net(output_iter.cuda()) + input_batch.cuda()
 output_2 = input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_1))
 output_3 = input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_2))
 return input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_3))



def implicit_iterations(net,input_batch,output,num_iter):

    output=output.cuda()
    iter=0
    while(iter < num_iter):
      output1 = (PEC4step_implicit(net,input_batch.cuda(),output)).cuda()
      print('residue inside implicit',torch.norm(output1-output))
      output = output1
      iter=iter+1 
    return output1


mynet = MLP_Net(input_size, hidden_layer_size, output_size).cuda()
mynet.load_state_dict(torch.load(net_file_name))
mynet.cuda()
print('Model loaded')

M = 99999
net_pred = np.zeros([M,np.size(label_test,1)])


for k in range(0,M):
 
    if (k==0):

        net_output =(PEC4step(mynet,input_test_torch[0,:]))
        net_output = implicit_iterations(mynet,input_test_torch[0,:].cuda(),net_output,lead)
        net_pred [k,:] = net_output.detach().cpu().numpy()

    else:

        net_output = (PEC4step(mynet,torch.from_numpy(net_pred[k-1,:]).float().cuda()))
        net_output = implicit_iterations(mynet,torch.from_numpy(net_pred[k-1,:]).float().cuda(),net_output,lead)
        net_pred [k,:] = net_output.detach().cpu().numpy()

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