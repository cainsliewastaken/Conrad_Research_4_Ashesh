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

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/'

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()

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
 output_1 = time_step*net(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))



################################################################
#  1d Fourier Integral Operator
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, time_future, time_history):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: a driving function observed at T timesteps + 1 locations (u(1, x), ..., u(T, x),  x).
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.modes = modes
        self.width = width
        self.time_future = time_future
        self.time_history = time_history
        self.fc0 = nn.Linear(self.time_history+1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

    def forward(self, u):
        grid = self.get_grid(u.shape, u.device)
        x = torch.cat((u, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 64 # input and output chasnnels to the FNO layer

learning_rate = 0.0001
lr_decay = 0.4
num_workers = 0  #What does this do?



# declare all 3 networks
mynet_directstep = FNO1d(modes, width, time_future, time_history).cuda()
mynet_Eulerstep = FNO1d(modes, width, time_future, time_history).cuda()
mynet_PECstep = FNO1d(modes, width, time_future, time_history).cuda()

#count_parameters(mynet_directstep)
mynet_directstep.load_state_dict(torch.load('NN_Spectral_Loss_FNO_Directstep_tendency_lambda_reg5lead1.pt'))
mynet_directstep.cuda()

#count_parameters(mynet_Eulerstep)
mynet_Eulerstep.load_state_dict(torch.load('NN_Spectral_Loss_FNO_Eulerstep_tendency_lambda_reg5lead1.pt'))
mynet_Eulerstep.cuda()

#count_parameters(mynet_PECstep)
mynet_PECstep.cuda()
mynet_PECstep.load_state_dict(torch.load('NN_Spectral_Loss_FNO_PECstep_tendency_lambda_reg5lead1.pt'))




M=99999
pred_direct = np.zeros([M,np.size(label_test,1)])
pred_Euler = np.zeros([M,np.size(label_test,1)])
pred_PEC = np.zeros([M,np.size(label_test,1)])

print('Eval started')

for k in range(0,M):
 
    if (k==0):

        out_direct = directstep(mynet_directstep,torch.reshape(input_test_torch[0,:],(1,input_size,1)))
        pred_direct [k,:] = torch.reshape(out_direct,(1,input_size)).detach().cpu().numpy()

        out_Euler = Eulerstep(mynet_Eulerstep,torch.reshape(input_test_torch[0,:],(1,input_size,1)))
        pred_Euler [k,:] = torch.reshape(out_Euler,(1,input_size)).detach().cpu().numpy()

        out_PEC = PECstep(mynet_PECstep,torch.reshape(input_test_torch[0,:],(1,input_size,1)))
        pred_PEC [k,:] = torch.reshape(out_PEC,(1,input_size)).detach().cpu().numpy()

    else:

        out_direct = directstep(mynet_directstep,torch.reshape(torch.from_numpy(pred_direct[k-1,:]),(1,input_size,1)).float().cuda())
        pred_direct [k,:] = torch.reshape(out_direct,(1,input_size)).detach().cpu().numpy()

        out_Euler = Eulerstep(mynet_Eulerstep,torch.reshape(torch.from_numpy(pred_Euler[k-1,:]),(1,input_size,1)).float().cuda())
        pred_Euler [k,:] = torch.reshape(out_Euler,(1,input_size)).detach().cpu().numpy()

        out_PEC = PECstep(mynet_PECstep,torch.reshape(torch.from_numpy(pred_PEC[k-1,:]),(1,input_size,1)).float().cuda())
        pred_PEC [k,:] = torch.reshape(out_PEC,(1,input_size)).detach().cpu().numpy()
    if k%1000==0:
       print(k)


def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

print('Eval finished')

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

print('calculations finished')


matfiledata_direct = {}
matfiledata_direct[u'prediction'] = pred_direct
matfiledata_direct[u'Truth'] = label_test 
matfiledata_direct[u'RMSE'] = RMSE(pred_direct, label_test)
matfiledata_direct[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_direct[u'pred_FFT'] = direct_1d_fspec_tdim
scipy.io.savemat(path_outputs+'predicted_directstep_1024_FNO_tendency_lead'+str(lead)+'.mat', matfiledata_direct)
#hdf5storage.write(matfiledata_direct, '.', path_outputs+'predicted_directstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)
print('Saved Direct prediction')


matfiledata_Euler = {}
matfiledata_Euler[u'prediction'] = pred_Euler
matfiledata_Euler[u'Truth'] = label_test 
matfiledata_Euler[u'RMSE'] = RMSE(pred_Euler, label_test)
scipy.io.savemat(path_outputs+'predicted_Eulerstep_1024_FNO_tendency_lead'+str(lead)+'.mat', matfiledata_Euler)
#hdf5storage.write(matfiledata_Euler, '.', path_outputs+'predicted_Eulerstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)
print('Saved Euler prediction')

matfiledata_PEC = {}
matfiledata_PEC[u'prediction'] = pred_PEC
matfiledata_PEC[u'Truth'] = label_test 
matfiledata_PEC[u'RMSE'] = RMSE(pred_PEC, label_test)
matfiledata_PEC[u'Truth_FFT'] = u_1d_fspec_tdim
matfiledata_PEC[u'pred_FFT'] = PEC_1d_fspec_tdim
scipy.io.savemat(path_outputs+'predicted_PECstep_1024_FNO_tendency_lead'+str(lead)+'.mat', matfiledata_PEC)
#hdf5storage.write(matfiledata_PEC, '.', path_outputs+'predicted_PECstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)
print('Saved PEC prediction')

print('Saved predictions, etc')



# create first plot
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(matfiledata_direct[u'RMSE'], label='Direct step')
ax1.plot(matfiledata_Euler[u'RMSE'], label='Euler step')
ax1.plot(matfiledata_PEC[u'RMSE'], label='PEC step')
ax1.set_xlabel('Time step')
ax1.set_ylabel('RSME')
ax1.legend(fontsize='x-small')
ax1.set(ylim=(0,30))
fig1.savefig(path_outputs+'RMSE.png')
print(sum(matfiledata_direct[u'RMSE']))


# fig12, ax12 = plt.subplots(figsize=(10,8))
# ax12.plot(matfiledata_direct[u'RMSE'], label='Direct step')
# ax12.legend(fontsize='x-small')
# # fig12.savefig('RMSE_direct.png')


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



        