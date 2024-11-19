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
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from ensemble_FP import ensemble_FP, ensemble_FP_cov
import hdf5storage

skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead,'lead')

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/MLP_PEC4step_lead1/chkpt_MLP_PEC4step_lead1_epoch60.pt"
#change this to use a different network
print(net_file_name)

step_func = PEC4step #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

num_ensembles = 100
noise_var = 10.0
wavenum_cutoff = 100

print('Noise ', noise_var)
print('Ensembles  ', num_ensembles)
print('Wave cut off ',wavenum_cutoff)

eval_output_name = 'PEC4_MLP_noise_'+str(noise_var)+'_'+str(num_ensembles)+'_ens_'+str(wavenum_cutoff)+'_wavenum_no_FP_short_t'
print(eval_output_name)

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/FP_output/'+eval_output_name+'/' #this is where the saved graphs and .mat files end up

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f: #change based on eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN=150000 #dont explicitly need this as no training is done in file
input_size = 1024
output_size = 1024

hidden_layer_size = 2000
num_layers = 6

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 512  # input and output chasnnels to the FNO layer

# my_net = FNO1d(modes, width, time_future, time_history)
my_net = MLP_Net(input_size, hidden_layer_size, output_size)

my_net.load_state_dict(torch.load(net_file_name))
my_net.cuda()
my_net.eval()



M = int(np.floor(999/lead))
net_pred = np.zeros([M,num_ensembles,np.size(label_test,1)])

print('Model loaded')

noised_input = (noise_var)*torch.randn(num_ensembles, 1024).cuda()
noised_input = label_test_torch[0,:].cuda() + noised_input

observation_func = lambda input: torch.fft.rfft(input, dim=1)[:,0:wavenum_cutoff].real.cuda() #input is ens by x_dim

y_vals_test = torch.zeros(label_test_torch.shape[0],wavenum_cutoff)
for i in range(label_test_torch.shape[0]):
    y_vals_test[i] = observation_func(label_test_torch[i,:].reshape(1, 1024))

y_true_invariant_mean = torch.mean(y_vals_test, dim=0).reshape(1, wavenum_cutoff)
y_true_invariant_std = torch.std(y_vals_test, dim=0).reshape(1, wavenum_cutoff)
# y_true_cov = torch.cov(y_vals_test.transpose(0,1))
# print(y_true_cov.shape)

with torch.no_grad():
    for k in range(0,M):
    #  print('time step',k)
        if (k==0):
            output = step_func(my_net, noised_input.reshape(num_ensembles,1024,1) ,time_step)
            net_pred[k,:,:,] = output.reshape(num_ensembles, 1024).detach().cpu().numpy()
        else:
            if k%1000==0:
                print(k)

            if (k>100) and (k%5==0):
                output = step_func(my_net, torch.from_numpy(net_pred[k-1,:,:,]).reshape(num_ensembles,1024,1).float().cuda(), time_step)
                output = ensemble_FP(output, observation_func, y_true_invariant_mean.cuda(), y_true_invariant_std.cuda())
                # output = ensemble_FP_cov(output, observation_func, y_true_invariant_mean.cuda(), y_true_cov.cuda())

                net_pred[k,:,:] = output.reshape(num_ensembles, 1024).detach().cpu().numpy()
            else:
                output = step_func(my_net, torch.from_numpy(net_pred[k-1,:,:,]).reshape(num_ensembles,1024,1).float().cuda(), time_step)
                net_pred[k,:,:] = output.reshape(num_ensembles, 1024).detach().cpu().numpy()

print('Eval Finished')
# print(net_pred.shape)
# temp_dict = {}
# temp_dict[u'prediction'] = net_pred
# temp_dict[u'truth'] = label_test[:net_pred.shape[0]]
# # scipy.io.savemat(path_outputs+eval_output_name+'_full_state.mat', temp_dict)
# # np.savez(path_outputs+eval_output_name+'_full_state', temp_dict)

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

def calc_save_chunk(net_pred_chunk, label_test_chunk, chunk_num):
    pred_RMSE = np.zeros([net_pred_chunk.shape[0],num_ensembles])
    # truth_fspec_x = np.zeros(np.shape(net_pred_chunk[:,:]), dtype=complex)
    net_pred_chunk_fspec_x = np.zeros(np.shape(net_pred_chunk[:,:,:]), dtype=complex)
    # truth_dt = np.diff(label_test_chunk, n=1, axis=0)
    # net_pred_chunk_dt = np.diff(net_pred_chunk, n=1, axis=0)
    # truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
    # net_pred_chunk_fspec_dt = np.zeros(np.shape(net_pred_chunk_dt[:,:,:]), dtype=complex)

    for ens in range(0,num_ensembles):
        #this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes

        pred_RMSE[:,ens] = RMSE(net_pred_chunk[:,ens,:], label_test_chunk[0:net_pred_chunk.shape[0]]).reshape(-1)

        for n in range(np.shape(net_pred_chunk)[0]):
            # truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test_chunk[n,:])) 
            net_pred_chunk_fspec_x[n, ens ,:] = np.abs(np.fft.fft(net_pred_chunk[n, ens, :])) 

        # for n in range(np.shape(net_pred_chunk_dt)[0]):
        #     # truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
        #     net_pred_chunk_fspec_dt[n, ens, :] = np.abs(np.fft.fft(net_pred_chunk_dt[n, ens, :])) 

    print('Calculation Finished')

    matfiledata_output = {}
    matfiledata_output[u'prediction'] = net_pred_chunk
    # matfiledata_output[u'Truth'] = label_test_chunk
    matfiledata_output[u'RMSE'] = pred_RMSE
    # matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
    matfiledata_output[u'pred_FFT_x'] = net_pred_chunk_fspec_x
    # matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
    # matfiledata_output[u'pred_FFT_dt'] = net_pred_chunk_fspec_dt
    # hdf5storage.write(matfiledata_output, '.',path_outputs+eval_output_name+'.mat', matlab_compatible=True)

    print('First save done')
    temp_matfile = {}
    temp_matfile[u'RMSE'] = matfiledata_output[u'RMSE']
    # scipy.io.savemat(path_outputs+eval_output_name+'_RMSE_chunk_'+str(chunk_num)+'.mat', temp_matfile)
    np.save(path_outputs+eval_output_name+'_RMSE_chunk_'+str(chunk_num), temp_matfile)

    # hdf5storage.write(temp_matfile, '.',path_outputs+eval_output_name+'_RMSE.mat', matlab_compatible=True)

    print('Saved main file')

    if skip_factor: #check if not == 0
        matfiledata_output_skip = {}
        matfiledata_output_skip[u'prediction'] = net_pred_chunk[0::skip_factor,:,:]
        # matfiledata_output_skip[u'Truth'] = label_test_chunk[0::skip_factor,:]
        matfiledata_output_skip[u'RMSE'] = pred_RMSE[0::skip_factor,:]
        # matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
        matfiledata_output_skip[u'pred_FFT_x'] = net_pred_chunk_fspec_x[0::skip_factor,:,:]
        # matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
        # matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_chunk_fspec_dt[0::skip_factor,:,:]
        
        # hdf5storage.write(matfiledata_output_skip, '.',path_outputs+eval_output_name+'_skip'+str(skip_factor)+'.mat', matlab_compatible=True)
        # scipy.io.savemat(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'_chunk_'+str(chunk_num)+'.mat', matfiledata_output_skip)
        np.save(path_outputs+eval_output_name+'_skip'+str(skip_factor)+'_chunk_'+str(chunk_num), matfiledata_output_skip)

prev_ind = 0
chunk_count = 0
num_chunks = 100
for chunk in np.array_split(net_pred, num_chunks):
    current_ind = prev_ind + chunk.shape[0]
    calc_save_chunk(chunk, label_test[prev_ind:current_ind], chunk_count)
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')

