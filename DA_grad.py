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
from nn_MLP import MLP_Net, MLP_net_variable
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from ensemble_FP import *
from Obs_func_generator import EnsKFstep_module
from nn_FP_net import *
from nn_AE import AE_Net
import hdf5storage

skip_factor = 100 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead,'lead')

net_file_name = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_lead1_tendency/chkpt_FNO_Eulerstep_lead1_tendency_epoch60.pt"
#change this to use a different network
print(net_file_name)

step_func = Eulerstep #this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

num_ensembles = 1
noise_var = 0
wave_num_start = 0
wavenum_cutoff = 512
# noise_cutoff = wavenum_cutoff

print('Noise ', noise_var)
print('Ensembles  ', num_ensembles)
print('Wave cut off ',wavenum_cutoff)

eval_output_name = 'Euler_FNO_tendency_noise_'+str(noise_var)+'_'+str(num_ensembles)+'_ens_DA_fft'
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
noised_input = ((noise_var)*torch.randn_like(input_test_torch) + input_test_torch)

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 256  # input and output chasnnels to the FNO layer

my_net = FNO1d(modes, width, time_future, time_history)
# my_net = MLP_Net(input_size, hidden_layer_size, output_size)

my_net.load_state_dict(torch.load(net_file_name))
my_net.cuda()
my_net.eval()

M = int(np.floor((99999)-1/lead))
# M = int(np.floor(9999/lead))

net_pred = np.zeros([M,num_ensembles, np.size(label_test,1)])

print('Model loaded')

observation_func_fft = lambda input: torch.fft.rfft(input, dim=1)[:,wave_num_start:wavenum_cutoff].cuda() #input is ens by x_dim

y_vals_test = torch.zeros(label_test_torch.shape[0],wavenum_cutoff - wave_num_start)
y_vals_test = observation_func_fft(label_test_torch.reshape(label_test_torch.shape[0], 1024))

y_true_invariant_mean = torch.mean(y_vals_test, dim=0).reshape(1, wavenum_cutoff - wave_num_start)
# y_true_invariant_mean[noise_cutoff - wave_num_start:] = 0

y_true_invariant_std = torch.std(y_vals_test, dim=0).reshape(1, wavenum_cutoff - wave_num_start)
# y_true_invariant_std[noise_cutoff - wave_num_start:] = 0

# y_true_cov = torch.cov(y_vals_test.transpose(0,1))

loss_func_fft = lambda input: torch.mean((y_true_invariant_mean   - observation_func_fft(input)).abs()) # 

        # AE_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/MLP_AE_unit_ball_w_kde_loss_w_origin/chkpt_MLP_AE_unit_ball_w_kde_loss_w_origin_epoch200.pt"
        # print(AE_file_path)

        # use_drop = False
        # latent_size = 10

        # my_net_AE_encoder = MLP_net_variable(input_size, latent_size, hidden_layer_size, 2, use_act=False, use_dropout=use_drop)
        # my_net_AE_decoder = MLP_net_variable(latent_size, output_size, hidden_layer_size, 4, use_act=False, use_dropout=use_drop)
        # my_net_AE = AE_Net(my_net_AE_encoder, my_net_AE_decoder)
        # my_net_AE.load_state_dict(torch.load(AE_file_path))

        # my_net_AE.cuda()
        # my_net_AE.eval()

        # observation_func_AE = lambda input: my_net_AE.encode(input.reshape(num_ensembles, input_size))

        # def loss_func_AE(input):
        #     obs_out = observation_func_AE(input)
        #     obs_out_normed = torch.nn.functional.normalize(obs_out, p=2.0, dim=1, eps=1e-12)
        #     return (((obs_out_normed.detach() - obs_out).pow(2)).mean()).pow(1)


# def differentiable_histogram_kde(data, n_bins=50, bandwidth=None):
#     """
#     Computes a differentiable histogram using kernel density estimation.

#     Args:
#         data (torch.Tensor): Input data.
#         n_bins (int): Number of bins in the histogram.
#         bandwidth (float, optional): Bandwidth of the kernel. If None, it is estimated from the data.

#     Returns:
#         torch.Tensor: Differentiable histogram.
#     """

#     data_min = noised_input.min().detach()
#     data_max = noised_input.max().detach()
#     bin_edges = torch.linspace(data_min, data_max, n_bins + 1).to(data.device)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     if bandwidth is None:
#       bandwidth = (data_max - data_min) / n_bins

#     histogram = torch.exp(-0.5 * ((bin_centers.unsqueeze(-1) - data.unsqueeze(0)) / bandwidth)**2).sum(-1)

#     return histogram / histogram.sum()

# print('Finding invarient measure')
# with torch.no_grad():
#     train_data_hist = differentiable_histogram_kde(label_test_torch.flatten()).detach().cuda()

# observation_func_hist = lambda input: differentiable_histogram_kde(input.flatten())

# loss_func_hist = lambda input: torch.sum((train_data_hist - observation_func_hist(input))**2)

# print('Invariant measure found')

# loss_func = lambda input: loss_func_fft(input) + loss_func_AE(input) + loss_func_hist(input)

def loss_func(input):
    loss1 = loss_func_fft(input) 
    # loss2 = loss_func_AE(input)
    # loss3 = loss_func_hist(input)
    # print(float(loss2))
    return (loss1) #(1*loss2) #torch.log(loss1/100)# +  #+ 0*loss3


noised_input = (noise_var)*torch.randn(num_ensembles, 1024).cuda()
noised_input = label_test_torch[0,:].cuda() + noised_input

num_backprop_steps = 1
num_iterations = 1
gamma = 0.75
last_grad = torch.zeros([num_ensembles, input_size, 1]).cuda()
# print(data.max(), data.min(), np.linalg.norm(data, ord=2, axis=0).mean(0))


for k in range(0,M):
#  print('time step',k)
    if (k==0):
        output = step_func(my_net, noised_input.reshape(num_ensembles,1024,1) ,time_step)
        
        net_pred[k,:,:,] = output.reshape(num_ensembles, 1024).detach().cpu().numpy()
        print(k, net_pred[k].max(), net_pred[k].min(), float(output.std(0).mean()))

    else:
        if (k%2000==0):
            print(k, net_pred[k-1].max(), net_pred[k-1].min(), net_pred[k-1].mean(0).max(), net_pred[k-1].mean(0).min())
            print(np.sqrt(np.mean((net_pred[k-1,0] -  label_test[k])**2)), float(loss))

        if ((k % num_backprop_steps == 0)):   # and (k >= 1000)
            # net_pred_temp_start = torch.tensor(net_pred[k-1], requires_grad=True).float().cuda()
            # net_pred_temp_start.retain_grad()

            # if num_backprop_steps > 1:
            #     net_pred_temp = torch.zeros([num_backprop_steps-1, num_ensembles, np.size(label_test,1), 1], requires_grad=False).cuda()
            #     net_pred_temp[0] = step_func(my_net, net_pred_temp_start.reshape(num_ensembles,1024,1), time_step)
            #     for i in range(1,num_backprop_steps-1):
            #         net_pred_temp[i] = step_func(my_net, net_pred_temp[i-1].reshape(num_ensembles,1024,1), time_step)
            #     loss = loss_func(net_pred_temp[-1])
                
            # else:
            #     loss = loss_func(step_func(my_net, net_pred_temp_start.reshape(num_ensembles,1024,1), time_step))
            # # print(loss)
            # # if loss>=0.0001:
            # #     print(k)
            # loss.backward()

            # current_grad = 1e-0*net_pred_temp_start.grad.reshape(num_ensembles, input_size, 1) + gamma * last_grad
            # new_input = torch.tensor(net_pred[k-1], requires_grad=False).reshape(num_ensembles,1024,1).float().cuda() - current_grad
            # last_grad = current_grad
            # output = step_func(my_net, new_input.reshape(num_ensembles,1024,1), time_step)
            
            output = step_func(my_net, torch.tensor(net_pred[k-1], requires_grad=True).reshape(num_ensembles,1024,1).float().cuda(), time_step)  
            # last_grad = 0

            # output.retain_grad()
            # loss = loss_func(output)
            # i=0
            # while (loss > 1e-3):
            #     loss.backward(retain_graph=True)
                
            #     current_grad = 1e-0*output.grad + gamma * last_grad
            #     output = output - current_grad
            #     last_grad = current_grad

            #     output.retain_grad()
            #     loss = loss_func(output)
                # i += 1
            # print(float(output[0,0]))

            for i in range(num_iterations):
                output.retain_grad()
                loss = loss_func(output)
                loss.backward(retain_graph=True)
                
                current_grad = 1e-3*output.grad + gamma * last_grad
                output = output - current_grad
                last_grad = current_grad

            net_pred[k,:,:] = output.reshape(num_ensembles, 1024).detach().cpu().numpy()

        else:
            output = step_func(my_net, torch.from_numpy(net_pred[k-1,:,:,]).reshape(num_ensembles,1024,1).float().cuda(), time_step)
            net_pred[k,:,:] = output.reshape(num_ensembles, 1024).detach().cpu().numpy()

    # print(np.sqrt(np.mean((net_pred[k-1,:] -  label_test[k-1])**2)), float(output.std(0).mean()))

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
    calc_save_chunk(chunk, label_test[prev_ind+1:current_ind+1], chunk_count)
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')
