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
#import hdf5storage
import pickle
import matplotlib.pyplot as plt




path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


def RK4step(net,input_batch):
 output_1 = net(input_batch.cuda())
 output_2 = net(input_batch.cuda()+0.5*output_1)
 output_3 = net(input_batch.cuda()+0.5*output_2)
 output_4 = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1 = net(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 
  

def directstep(net,input_batch):
  output_1 = net(input_batch.cuda())
  return output_1

def PECstep(net,input_batch):
 output_1 = net(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))

def PEC4step(net,input_batch):
 output_1 = time_step*net(input_batch.cuda()) + input_batch.cuda()
 output_2 = input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))
 output_3 = input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_2))
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_3))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il  = ((nn.Linear(input_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.il.weight)

        self.hidden1  = ((nn.Linear(hidden_layer_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.hidden1.weight)

        self.hidden2  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden2.weight)

        self.hidden3  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden3.weight)

        self.hidden4  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden4.weight)

        self.hidden5  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden5.weight)        

        self.ol  = nn.Linear(hidden_layer_size,output_size)
        torch.nn.init.xavier_uniform_(self.ol.weight)

        self.tanh = nn.Tanh()


    def forward(self,x):
        
        x1 = self.tanh(self.il(x))
        x2 = self.tanh(self.hidden1(x1))
        x3 = self.tanh(self.hidden2(x2))
        x4 = self.tanh(self.hidden3(x3))
        x5 = self.tanh(self.hidden4(x4))
        x6 = self.tanh(self.hidden5(x5))
        out =self.ol(x6)
        return out





mynet_directstep = Net()
mynet_directstep.load_state_dict(torch.load('NN_directstep_lead1.pt'))

mynet_Eulerstep = Net()
mynet_Eulerstep.load_state_dict(torch.load('NN_Eulerstep_lead1.pt'))

mynet_RK4step = Net()
mynet_RK4step.load_state_dict(torch.load('NN_RK4step_lead1.pt'))

mynet_PECstep = Net()
mynet_PECstep.load_state_dict(torch.load('NN_PECstep_lead1.pt'))

mynet_directstep.cuda()

mynet_Eulerstep.cuda()

mynet_RK4step.cuda()

mynet_PECstep.cuda()




M=99999
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

        out_PEC = PECstep(mynet_PECstep,input_test_torch[0,:])
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
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 



#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
u_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)
direct_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)
PEC_1d_fspec_tdim = np.zeros(np.shape(label_test[:,:]), dtype=complex)

for n in range(np.shape(label_test)[0]):
    u_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    direct_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(pred_direct[n,:])) 
    PEC_1d_fspec_tdim[n,:] = np.abs(np.fft.fft(pred_PEC[n,:])) 


# calculate time derivative using 2nd order finite diff
u_truth_difft_n2 = np.diff(label_test, n=2, axis=0)
u_direct_difft_n2 = np.diff(pred_direct, n=2, axis=0)
u_PEC_difft_n2 = np.diff(pred_PEC, n=2, axis=0)

# calculate fourier spectrum of time derivative along a single timestep
u_truth_difft_n2_fspec = np.zeros(np.shape(u_truth_difft_n2[:,:]), dtype=complex)
u_direct_difft_n2_fspec = np.zeros(np.shape(u_direct_difft_n2[:,:]), dtype=complex)
u_PEC_difft_n2_fspec = np.zeros(np.shape(u_PEC_difft_n2[:,:]), dtype=complex)


for n in range(np.shape(u_truth_difft_n2)[0]):
    u_truth_difft_n2_fspec[n,:] = np.abs(np.fft.fft(u_truth_difft_n2[n,:])) 
    u_direct_difft_n2_fspec[n,:] = np.abs(np.fft.fft(u_direct_difft_n2[n,:])) 
    u_PEC_difft_n2_fspec[n,:] = np.abs(np.fft.fft(u_PEC_difft_n2[n,:])) 



matfiledata_direct = {}
matfiledata_direct[u'prediction'] = pred_direct
matfiledata_direct[u'Truth'] = label_test 
matfiledata_direct[u'RMSE'] = RMSE(pred_direct, label_test)
matfiledata_direct[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_direct[u'pred_FFT'] = direct_1d_fspec_tdim
scipy.io.savemat(path_outputs+'predicted_directstep_1024_lead'+str(lead)+'.mat', matfiledata_direct)
#hdf5storage.write(matfiledata_direct, '.', path_outputs+'predicted_directstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

matfiledata_Euler = {}
matfiledata_Euler[u'prediction'] = pred_Euler
matfiledata_Euler[u'Truth'] = label_test 
matfiledata_Euler[u'RMSE'] = RMSE(pred_Euler, label_test)
scipy.io.savemat(path_outputs+'predicted_Eulerstep_1024_lead'+str(lead)+'.mat', matfiledata_Euler)
#hdf5storage.write(matfiledata_Euler, '.', path_outputs+'predicted_Eulerstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

matfiledata_RK4 = {}
matfiledata_RK4[u'prediction'] = pred_RK4
matfiledata_RK4[u'Truth'] = label_test 
matfiledata_RK4[u'RMSE'] = RMSE(pred_RK4, label_test)
scipy.io.savemat(path_outputs+'predicted_RK4step_1024_lead'+str(lead)+'.mat', matfiledata_RK4)
#hdf5storage.write(matfiledata_RK4, '.', path_outputs+'predicted_RK4step_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

matfiledata_PEC = {}
matfiledata_PEC[u'prediction'] = pred_PEC
matfiledata_PEC[u'Truth'] = label_test 
matfiledata_PEC[u'RMSE'] = RMSE(pred_PEC, label_test)
matfiledata_PEC[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_PEC[u'pred_FFT'] = PEC_1d_fspec_tdim
scipy.io.savemat(path_outputs+'predicted_PECstep_1024_lead'+str(lead)+'.mat', matfiledata_PEC)
#hdf5storage.write(matfiledata_PEC, '.', path_outputs+'predicted_PECstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved predictions, etc')



# create first plot
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(matfiledata_direct[u'RMSE'], label='Direct step')
ax1.plot(matfiledata_Euler[u'RMSE'], label='Euler step')
ax1.plot(matfiledata_RK4[u'RMSE'], label='RK4 step')
ax1.plot(matfiledata_PEC[u'RMSE'], label='PEC step')
ax1.set_xlabel('Time step')
ax1.set_ylabel('RSME')
ax1.legend(fontsize='x-small')
fig1.savefig(path_outputs+'RMSE.png')
#print(sum(matfiledata_direct[u'RMSE']))

fig12, ax12 = plt.subplots(figsize=(10,8))
ax12.plot(matfiledata_direct[u'RMSE'], label='Direct step')
ax12.legend(fontsize='x-small')
# fig12.savefig('RMSE_direct.png')


# create second plot
fig2, ax2 = plt.subplots(2,2, figsize=(10,8))
ax2[0,0].plot(pred_PEC[1,:], label='PEC step')
ax2[0,0].plot(label_test[1,:], label='Truth')
ax2[0,0].plot(pred_direct[1,:], label='Direct step')
ax2[0,0].set_title('t = 1')
ax2[0,0].legend(fontsize='x-small')
#ax2[0,0].set_xlabel('Spacial grid')
ax2[0,0].set_ylabel('Values of u(x)')
ax2[0,0].set(ylim=(-5,5))


ax2[0,1].plot(pred_PEC[10000,:], label='PEC step')
ax2[0,1].plot(label_test[10000,:], label='Truth')
ax2[0,1].plot(pred_direct[10000,:], label='Direct step')
ax2[0,1].set_title('t = 10000')
ax2[0,1].legend(fontsize='x-small')
#ax2[0,1].set_xlabel('Spacial grid')
ax2[0,1].set_ylabel('Values of u(x)')
ax2[0,1].set(ylim=(-5,5))


ax2[1,0].plot(pred_PEC[20000,:], label='PEC step')
ax2[1,0].plot(label_test[20000,:], label='Truth')
ax2[1,0].plot(pred_direct[20000,:], label='Direct step')
ax2[1,0].set_title('t = 20000')
ax2[1,0].legend(fontsize='x-small')
ax2[1,0].set_xlabel('Spacial grid')
ax2[1,0].set_ylabel('Values of u(x)')
ax2[1,0].set(ylim=(-5,5))


ax2[1,1].plot(pred_PEC[-1,:], label='PEC step')
ax2[1,1].plot(label_test[-1,:], label='Truth')
ax2[1,1].plot(pred_direct[-1,:], label='Direct step')
ax2[1,1].set_title('t = 100000')
ax2[1,1].legend(fontsize='x-small')
ax2[1,1].set_xlabel('Spacial grid')
ax2[1,1].set_ylabel('Values of u(x)')
ax2[1,1].set(ylim=(-5,5))


fig2.savefig(path_outputs+'Values_at_multiple_timesteps.png')


# create third plot, fspec at multiple timesteps
fig3, ax3 = plt.subplots(2,1, figsize=(10,8))
ax3[0].loglog(u_1d_fspec_tdim[1,1:512], label='Truth at t=1')
# ax3[0].loglog(u_1d_fspec_tdim[10000,1:512], label='Truth at t=10000')
# ax3[0].loglog(u_1d_fspec_tdim[20000,1:512], label='Truth at t=20000')
# ax3[0].loglog(u_1d_fspec_tdim[-1,1:512], label='Truth at t=100000')
ax3[0].loglog(direct_1d_fspec_tdim[1,1:512], label='Direct step at t=1')
ax3[0].loglog(direct_1d_fspec_tdim[500,1:512], label='Direct step at t=500')
ax3[0].loglog(direct_1d_fspec_tdim[20000,1:512], label='Direct step at t=20000')
# ax3[0].loglog(direct_1d_fspec_tdim[-1,1:512], label='Direct step at t=100000')
ax3[0].legend(fontsize='x-small')
ax3[0].set_xlabel('Fourier modes')
ax3[0].set_ylabel('Amplitudes')
ax3[0].set_title('Values of the fourier spectrum of u and direct step', loc='left')


ax3[1].loglog(u_1d_fspec_tdim[1,1:512], label='Truth at t=1')
# ax3[1].loglog(u_1d_fspec_tdim[10000,1:512], label='Truth at t=10000')
# ax3[1].loglog(u_1d_fspec_tdim[20000,1:512], label='Truth at t=20000')
# ax3[1].loglog(u_1d_fspec_tdim[-1,1:512], label='Truth at t=100000')
ax3[1].loglog(PEC_1d_fspec_tdim[0,1:512], label='PEC step at t=1')
ax3[1].loglog(PEC_1d_fspec_tdim[500,1:512], label='PEC step at t=500')
ax3[1].loglog(PEC_1d_fspec_tdim[20000,1:512], label='PEC step at t=20000')
# ax3[1].loglog(PEC_1d_fspec_tdim[-1,1:512], label='PEC step at t=100000')
ax3[1].legend(fontsize='x-small')
ax3[1].set_title('Values of the fourier spectrum of u and PEC step', loc='left')
ax3[1].set_xlabel('Fourier modes')
ax3[1].set_ylabel('Amplitudes')
fig3.savefig(path_outputs+'Fspec_at_multiple_timesteps.png')


# create fourth plot, fspec of time derivative at multiple timesteps
fig4, ax4 = plt.subplots(2,1, figsize=(10,8))
ax4[0].loglog(u_truth_difft_n2_fspec[0,1:512], label='Truth at t=0')
# ax4[0].loglog(u_truth_difft_n2_fspec[10000,1:512], label='Truth at t=10000')
# ax4[0].loglog(u_truth_difft_n2_fspec[20000,1:512], label='Truth at t=20000')
# ax4[0].loglog(u_truth_difft_n2_fspec[-1,1:512], label='Truth at t=100000')
ax4[0].loglog(u_direct_difft_n2_fspec[0,1:512], label='Direct step at t=0')
ax4[0].loglog(u_direct_difft_n2_fspec[10000,1:512], label='Direct step at t=10000')
ax4[0].loglog(u_direct_difft_n2_fspec[20000,1:512], label='Direct step at t=20000')
ax4[0].loglog(u_direct_difft_n2_fspec[-1,1:512], label='Direct step at t=100000')
ax4[0].legend(fontsize='x-small')
ax4[0].set_title('Values of the fourier spectrum of du/dt and direct step time derivative at timesteps 0, 10k, 20k and 100k')
ax4[0].set_xlabel('Fourier modes')
ax4[0].set_ylabel('Amplitudes')


ax4[1].loglog(u_truth_difft_n2_fspec[0,1:512], label='Truth at t=0')
# ax4[1].loglog(u_truth_difft_n2_fspec[10000,1:512], label='Truth at t=10000')
# ax4[1].loglog(u_truth_difft_n2_fspec[20000,1:512], label='Truth at t=20000')
# ax4[1].loglog(u_truth_difft_n2_fspec[-1,1:512], label='Truth at t=100000')
ax4[1].loglog(u_PEC_difft_n2_fspec[0,1:512], label='PEC step at t=0')
ax4[1].loglog(u_PEC_difft_n2_fspec[10000,1:512], label='PEC step at t=10000')
ax4[1].loglog(u_PEC_difft_n2_fspec[20000,1:512], label='PEC step at t=20000')
ax4[1].loglog(u_PEC_difft_n2_fspec[-1,1:512], label='PEC step at t=100000')
ax4[1].legend(fontsize='x-small')
ax4[1].set_title('Values of the fourier spectrum of du/dt and PEC step time derivative at timesteps 0, 10k, 20k and 100k')
ax4[1].set_xlabel('Fourier modes')
ax4[1].set_ylabel('Amplitudes')

fig4.savefig(path_outputs+'Time_derivative_Fspec_at_multiple_timesteps.png')


print('Graphs plotted and saved')



        