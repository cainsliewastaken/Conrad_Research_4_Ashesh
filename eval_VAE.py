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
from nn_VAE import VAE

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead, "Lead")

num_ensambles = 10


skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/VAE_output/' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/Conrad_Research_4_Ashesh/VAE_RK4step_lead1.pt" #change this to use a different network

eval_output_name = 'predicted_RK4step_VAE_1024_lead'+str(lead)+''  # what to name the output file, .mat ending not needed

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f: #change for eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000 #dont explicitly need this as no training is done in file, here to help separate training data from eval data
input_size = 1024


input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float().cuda()
label_test = np.transpose(data[:,trainN+lead::lead])

#model parameters
imgChannels = 1
out_channels = 1
num_filters = 64
dimx = 1024
dimy = 1
zDim = 256
learning_rate = 0.001
lr_decay = 0.4

mynet = VAE(imgChannels, out_channels, num_filters, dimx, dimy, zDim).cuda()
mynet.load_state_dict(torch.load(net_file_name))

mynet.cuda()
print('Model loaded')

M = int(np.floor(99999/lead))
net_pred = np.zeros([M,num_ensambles,np.size(label_test,1)])
net_pred_mean_traj = np.zeros([M,np.size(label_test,1)])



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

step_func = RK4step
print(step_func,'Step function')

for k in range(0,M):
#  print('time step',k)
 if (k==0):

   for ens in range (0,num_ensambles):
    output = step_func(label_test_torch[0,:])
    net_pred[k,ens,:,] = np.squeeze(output.detach().cpu().numpy())

 else:

   mean_traj = torch.from_numpy(np.mean(net_pred[k-1,:,:],0)).float().cuda()
   net_pred_mean_traj[k-1,:] = mean_traj.detach().cpu().numpy()
   for ens in range (0, num_ensambles):
     output = step_func(mean_traj)
     net_pred[k,ens,:,] = np.squeeze(output.detach().cpu().numpy())

net_pred_mean_traj[-1,:] = torch.from_numpy(np.mean(net_pred[-1,:,:],0))

print('Eval Finished')

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
truth_fspec_x = np.zeros(np.shape(net_pred_mean_traj[:,:]), dtype=complex)
net_pred_fspec_x = np.zeros(np.shape(net_pred_mean_traj[:,:]), dtype=complex)

for n in range(np.shape(net_pred_mean_traj)[0]):
    truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    net_pred_fspec_x[n,:] = np.abs(np.fft.fft(net_pred_mean_traj[n,:])) 


# calculate time derivative using 1st order finite diff
truth_dt = np.diff(label_test, n=1, axis=0)
net_pred_dt = np.diff(net_pred_mean_traj, n=1, axis=0)

# calculate fourier spectrum of time derivative along a single timestep
truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
net_pred_fspec_dt = np.zeros(np.shape(net_pred_dt[:,:]), dtype=complex)


for n in range(np.shape(truth_dt)[0]):
    truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    net_pred_fspec_dt[n,:] = np.abs(np.fft.fft(net_pred_dt[n,:])) 



matfiledata_output = {}
matfiledata_output[u'prediction'] = net_pred_mean_traj
# matfiledata_output[u'pred_all_ensembles'] = net_pred
matfiledata_output[u'Truth'] = label_test 
matfiledata_output[u'RMSE'] = RMSE(net_pred_mean_traj, label_test)
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
    matfiledata_output_skip[u'prediction'] = net_pred_mean_traj[0::skip_factor,:]
    matfiledata_output_skip[u'pred_all_ensembles'] = net_pred[0::skip_factor,:]
    matfiledata_output_skip[u'Truth'] = label_test[0::skip_factor,:]
    matfiledata_output_skip[u'RMSE'] = RMSE(net_pred_mean_traj, label_test)[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_x'] = net_pred_fspec_x[0::skip_factor,:]
    matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
    matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_fspec_dt[0::skip_factor,:]

    scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matfiledata_output_skip)
print('Data saved')