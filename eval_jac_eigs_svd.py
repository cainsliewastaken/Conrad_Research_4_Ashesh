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



skip_factor = 0

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/Net_output_pred_jacs/' #this is where the saved graphs and .mat files end up

file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/Net_output_pred_jacs/KS_pred_Eulerstep_FNO_jacs_for_1k/KS_pred_Eulerstep_FNO_jacs_for_1k_chunk_"
print(file_path)
#change this to use a different network

eval_output_name = 'KS_pred_Eulerstep_FNO_eigs_svd_for_1k'  # what to name the output file, .mat ending not needed
print(eval_output_name)

M = 999
num_chunks = 100

def load_output_data_jacs(num_chunks, file_path, M):

    # output_data = np.zeros([int(M)+1,1024])
    # output_data_RMSE = np.zeros([int(M)+1])
    # output_data_FFT_X = np.zeros([int(M)+1,512])
    output_jacs = np.zeros([int(M)+1,1024,1024])
    output_jacs_true = np.zeros([int(M)+1,1024,1024])


    start_ind = 0
    for k in range(num_chunks):
        out = np.load(file_path+str(k)+'.npy', allow_pickle=True)#[10000:,:,:,:]
        t_range = out.item()['prediction'].shape[0]
        # output_data[start_ind:start_ind+t_range] = out.item()['prediction']
        # output_data_RMSE[start_ind:start_ind+t_range] = out.item()['RMSE']
        # output_data_FFT_X[start_ind:start_ind+t_range,:512] = out.item()['pred_FFT_x'][:,:512]
        output_jacs[start_ind:start_ind+t_range] = out.item()['Jacobians']
        output_jacs_true[start_ind:start_ind+t_range] = out.item()['Jacobians_truth']

        start_ind += t_range
        # print(start_ind)

    # return torch.from_numpy(output_data).float(), torch.from_numpy(output_data_RMSE).float(), torch.from_numpy(output_data_FFT_X).float(), torch.from_numpy(output_jacs).float(), torch.from_numpy(output_jacs_true).float()
    return torch.from_numpy(output_jacs).float().cuda(), torch.from_numpy(output_jacs_true).float().cuda()


net_jacs, net_jacs_true = load_output_data_jacs(num_chunks, file_path, M)

print('Beggining calculations')
with torch.no_grad():

    net_eigs, net_eigvecs = torch.linalg.eig(net_jacs.cuda())
    net_eigs_true, net_eigvecs_true = torch.linalg.eig(net_jacs.cuda())

    _, net_eigs_inds = torch.sort(net_eigs.abs(), 1, descending=True) #sorted largest to smallest
    net_eigs = net_eigs[net_eigs_inds]
    net_eigvecs = net_eigvecs[:,net_eigs_inds]

    _, net_eigs_inds_true = torch.sort(net_eigs_true.abs(), 1, descending=True) #sorted largest to smallest
    net_eigs_true = net_eigs_true[net_eigs_inds_true]

    net_eigvecs_true = net_eigvecs_true[:,net_eigs_inds_true]

    U, S, Vh = torch.linalg.svd(net_jacs, dim=1, full_matrices=False)
    U_true, S_true, Vh_true = torch.linalg.svd(net_jacs_true, dim=1, full_matrices=False)

print('Calculations finished')

def calc_save_chunk(chunk_num, net_eigs_chunk, net_eigvecs_chunk, net_eigs_true_chunk, net_eigvecs_true_chunk, U_chunk, S_chunk, Vh_chunk, U_true_chunk, S_true_chunk, Vh_true_chunk):
    
    matfiledata_output = {}
    matfiledata_output[u'net_eigs_chunk'] = net_eigs_chunk.cpu()
    matfiledata_output[u'net_eigvecs_chunk'] = net_eigvecs_chunk.cpu()
    matfiledata_output[u'net_eigs_true_chunk'] = net_eigs_true_chunk.cpu()
    matfiledata_output[u'net_eigvecs_true_chunk'] = net_eigvecs_true_chunk.cpu()
    matfiledata_output[u'U_chunk'] = U_chunk.cpu()
    matfiledata_output[u'S_chunk'] = S_chunk.cpu()
    matfiledata_output[u'Vh_chunk'] = Vh_chunk.cpu()
    matfiledata_output[u'U_true_chunk'] = U_true_chunk.cpu()
    matfiledata_output[u'S_true_chunk'] = S_true_chunk.cpu()
    matfiledata_output[u'Vh_true_chunk'] = Vh_true_chunk.cpu()

    np.save(path_outputs+'/'+eval_output_name+'/'+eval_output_name+'_chunk_'+str(chunk_num), matfiledata_output)


if not os.path.exists(path_outputs+'/'+eval_output_name+'/'):
    os.makedirs(path_outputs+'/'+eval_output_name+'/')
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' created.")
else:
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' already exists.")
print('Starting save')
prev_ind = 0
chunk_count = 0
num_chunks = 100
for chunk in np.array_split(range(net_eigs.shape[0]), num_chunks):
    current_ind = prev_ind + chunk.shape[0]
    calc_save_chunk(chunk_count, net_eigs[prev_ind:current_ind], net_eigvecs[prev_ind:current_ind],
                     net_eigs_true[prev_ind:current_ind], net_eigvecs_true[prev_ind:current_ind],
                     U[prev_ind:current_ind], S[prev_ind:current_ind], Vh[prev_ind:current_ind],
                     U_true[prev_ind:current_ind], S_true[prev_ind:current_ind], Vh_true[prev_ind:current_ind])
    
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')
