import numpy as np
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
import hdf5storage
import pickle
from nn_train_all4 import Net, directstep, Eulerstep, RK4step, PECstep

path_outputs = '/media/volume/sdb/conrad_temp/model_eval/'

with open('/media/volume/sdb/conrad_temp/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])



mynet_directstep = Net()
mynet_Eulerstep = Net()
mynet_RK4step = Net()
mynet_PECstep = Net()






M=100000
pred_direct = np.zeros([M,np.size(label_test,1)])
pred_Euler = np.zeros([M,np.size(label_test,1)])
pred_RK4 = np.zeros([M,np.size(label_test,1)])
pred_PEC = np.zeros([M,np.size(label_test,1)])


for k in range(0,M):
 
    if (k==0):

        out_direct = directstep(mynet_directstep,input_test_torch[0,:])
        pred_direct [k,:] = out_direct.detach().cpu().numpy()

        out_Euler = Eulerstep(mynet_Eulerstep,input_test_torch[0,:])
        pred_Euler [k,:] = out_Euler.detach().cpu().numpy()

        out_RK4 = RK4step(mynet_RK4step,input_test_torch[0,:])
        pred_RK4 [k,:] = out_RK4.detach().cpu().numpy()

        out_PEC = PECstep(mynet_RK4step,input_test_torch[0,:])
        pred_PEC [k,:] = out_PEC.detach().cpu().numpy()

    else:

        out_direct = directstep(mynet_directstep,torch.from_numpy(pred_direct[k-1,:]).float().cuda())
        pred_direct [k,:] = out_direct.detach().cpu().numpy()

        out_Euler = Eulerstep(mynet_Eulerstep,torch.from_numpy(pred_Euler[k-1,:]).float().cuda())
        pred_Euler [k,:] = out_Euler.detach().cpu().numpy()

        out_RK4 = RK4step(mynet_RK4step,torch.from_numpy(pred_RK4[k-1,:]).float().cuda())
        pred_RK4 [k,:] = out_RK4.detach().cpu().numpy()

        out_PEC = PECstep(mynet_PECstep,torch.from_numpy(pred_PEC[k-1,:]).float().cuda())
        pred_PEC [k,:] = out_PEC.detach().cpu().numpy()


def RMSE(y_hat, y_true):
    return torch.sqrt(torch.mean(y_hat - y_true)**2)



#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
u_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)
direct_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)
PEC_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)

for n in range(np.shape(label_test)[0]):
    u_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    direct_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(pred_direct[n,:])) 
    PEC_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(pred_PEC[n,:])) 


matfiledata_direct = {}
matfiledata_direct[u'prediction'] = pred_direct
matfiledata_direct[u'Truth'] = label_test 
matfiledata_direct[u'RMSE'] = RMSE(pred_direct, label_test)
matfiledata_direct[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_direct[u'pred_FFT'] = direct_1d_fspec_tdim
hdf5storage.write(matfiledata_direct, '.', path_outputs+'predicted_directstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

matfiledata_Euler = {}
matfiledata_Euler[u'prediction'] = pred_Euler
matfiledata_Euler[u'Truth'] = label_test 
matfiledata_Euler[u'RMSE'] = RMSE(pred_Euler, label_test)
hdf5storage.write(matfiledata_Euler, '.', path_outputs+'predicted_Eulerstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

matfiledata_RK4 = {}
matfiledata_RK4[u'prediction'] = pred_RK4
matfiledata_RK4[u'Truth'] = label_test 
matfiledata_RK4[u'RMSE'] = RMSE(pred_RK4, label_test)
hdf5storage.write(matfiledata_RK4, '.', path_outputs+'predicted_RK4step_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

matfiledata_PEC = {}
matfiledata_PEC[u'prediction'] = pred_PEC
matfiledata_PEC[u'Truth'] = label_test 
matfiledata_PEC[u'RMSE'] = RMSE(pred_PEC, label_test)
matfiledata_PEC[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_PEC[u'pred_FFT'] = PEC_1d_fspec_tdim
hdf5storage.write(matfiledata_PEC, '.', path_outputs+'predicted_PECstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved predictions, etc')




# dont need to linearize yet, going to do this later or in another file

# inputs_w_grad_direct = torch.zeros([int(4),input_size])
# inputs_w_grad_direct = torch.zeros([int(4),input_size])
# inputs_w_grad_direct = torch.zeros([int(4),input_size])


# i = 0
# for j in np.array(int([0, 10000, 50000, 100000])):
#     inputs_w_grad_direct[i,:] = pred_direct[j,:].requires_grad_(requires_grad=True)
#     i += 1





print('Saved Predictions')

        