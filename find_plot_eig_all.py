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






path_outputs = '/media/volume/sdb/conrad_temp/model_eval/'

with open('/media/volume/sdb/conrad_temp/training_data/KS_1024.pkl', 'rb') as f:
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


# Declare all needed network definitions. Prob should have in another file

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


#load basic model predictions
mynet_directstep = Net()
mynet_directstep.load_state_dict(torch.load('NN_directstep_lead1.pt'))
mynet_directstep.cuda()

mynet_Eulerstep = Net()
mynet_Eulerstep.load_state_dict(torch.load('NN_Eulerstep_lead1.pt'))
mynet_Eulerstep.cuda()

mynet_RK4step = Net()
mynet_RK4step.load_state_dict(torch.load('NN_RK4step_lead1.pt'))
mynet_RK4step.cuda()

mynet_PECstep = Net()
mynet_RK4step.load_state_dict(torch.load('NN_PECstep_lead1.pt'))
mynet_PECstep.cuda()


val_dict_direct = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval/predicted_directstep_1024_lead1.mat"')
val_dict_Euler = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval/predicted_Eulerstep_1024_lead1.mat"')
val_dict_RK4 = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval/predicted_RK4step_1024_lead1.mat"')
val_dict_PEC = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval/predicted_PECstep_1024_lead1.mat"')


pred_direct = val_dict_direct[u'prediction']
pred_Euler = val_dict_Euler[u'prediction']
pred_RK4 = val_dict_RK4[u'prediction']
pred_PEC = val_dict_PEC[u'prediction']


ygrad_direct = torch.zeros([int(4),input_size,input_size])
ygrad_Euler = torch.zeros([int(4),input_size,input_size])
ygrad_RK4 = torch.zeros([int(4),input_size,input_size])
ygrad_PEC = torch.zeros([int(4),input_size,input_size])



#tendency models and predictions
mynet_directstep_tendency = Net()
mynet_directstep_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_directstep_lead1.pt'))
mynet_directstep_tendency.cuda()

mynet_Eulerstep_tendency = Net()
mynet_Eulerstep_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_Eulerstep_lead1.pt'))
mynet_Eulerstep_tendency.cuda()

mynet_RK4step_tendency = Net()
mynet_RK4step_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_RK4step_lead1.pt'))
mynet_RK4step_tendency.cuda()

mynet_PECstep_tendency = Net()
mynet_RK4step_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_PECstep_lead1.pt'))
mynet_PECstep_tendency.cuda()

val_dict_direct_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_directstep_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_Euler_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_Eulerstep_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_RK4_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_RK4step_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_PEC_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_PECstep_1024_lead1_lambda_reg5_tendency.mat"')

pred_direct_tendency = val_dict_direct_tendency[u'prediction']
pred_Euler_tendency = val_dict_Euler_tendency[u'prediction']
pred_RK4_tendency = val_dict_RK4_tendency[u'prediction']
pred_PEC_tendency = val_dict_PEC_tendency[u'prediction']

ygrad_direct_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_RK4_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_tendency = torch.zeros([int(4),input_size,input_size])


# FNO models and predictions
mynet_directstep_FNO = Net()
mynet_directstep_FNO.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_directstep_lead1.pt'))
mynet_directstep_FNO.cuda()

mynet_Eulerstep_FNO = Net()
mynet_Eulerstep_FNO.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_Eulerstep_lead1.pt'))
mynet_Eulerstep_FNO.cuda()

mynet_PECstep_FNO = Net()
mynet_PECstep_FNO.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_PECstep_lead1.pt'))
mynet_PECstep_FNO.cuda()

val_dict_direct_FNO = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_directstep_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_Euler_FNO = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_Eulerstep_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_PEC_FNO = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_PECstep_1024_lead1_lambda_reg5_tendency.mat"')

pred_direct_FNO = val_dict_direct_FNO[u'prediction']
pred_Euler_FNO = val_dict_Euler_FNO[u'prediction']
pred_PEC_FNO = val_dict_PEC_FNO[u'prediction']

ygrad_direct_FNO = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_FNO = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_FNO = torch.zeros([int(4),input_size,input_size])


# FNO + tendency models and predictions
mynet_directstep_FNO_tendency = Net()
mynet_directstep_FNO_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_directstep_lead1.pt'))
mynet_directstep_FNO_tendency.cuda()

mynet_Eulerstep_FNO_tendency = Net()
mynet_Eulerstep_FNO_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_Eulerstep_lead1.pt'))
mynet_Eulerstep_FNO_tendency.cuda()

mynet_PECstep_FNO_tendency = Net()
mynet_PECstep_FNO_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_PECstep_lead1.pt'))
mynet_PECstep_FNO_tendency.cuda()

val_dict_direct_FNO_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_directstep_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_Euler_FNO_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_Eulerstep_1024_lead1_lambda_reg5_tendency.mat"')
val_dict_PEC_FNO_tendency = scipy.io.loadmat('"/media/volume/sdb/conrad_temp/model_eval_tendency/predicted_PECstep_1024_lead1_lambda_reg5_tendency.mat"')

pred_direct_FNO_tendency = val_dict_direct_FNO_tendency[u'prediction']
pred_Euler_FNO_tendency = val_dict_Euler_FNO_tendency[u'prediction']
pred_PEC_FNO_tendency = val_dict_PEC_FNO_tendency[u'prediction']

ygrad_direct_FNO_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_FNO_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_FNO_tendency = torch.zeros([int(4),input_size,input_size])


i = 0
for j in np.array(int([0, 10000, 50000, 100000])):
    #basic linear model jacobian calculation

    ygrad_direct[i,:,:] = torch.func.jacrev(directstep, argnums=1)(mynet_directstep, pred_direct[j,:])
    ygrad_Euler[i,:,:] = torch.func.jacrev(Eulerstep, argnums=1)(mynet_Eulerstep, pred_Euler[j,:])
    ygrad_RK4[i,:,:] = torch.func.jacrev(RK4step, argnums=1)(mynet_RK4step, pred_RK4[j,:])
    ygrad_PEC[i,:,:] = torch.func.jacrev(PECstep, argnums=1)(mynet_PECstep, pred_PEC[j,:])

    #linear plus tendency (spectral loss) jacobian calculation

    ygrad_direct_tendency[i,:,:] = torch.func.jacrev(directstep, argnums=1)(mynet_directstep_tendency, pred_direct_tendency[j,:])
    ygrad_Euler_tendency[i,:,:] = torch.func.jacrev(Eulerstep, argnums=1)(mynet_Eulerstep_tendency, pred_Euler_tendency[j,:])
    ygrad_RK4_tendency[i,:,:] = torch.func.jacrev(RK4step, argnums=1)(mynet_RK4step_tendency, pred_RK4_tendency[j,:])
    ygrad_PEC_tendency[i,:,:] = torch.func.jacrev(PECstep, argnums=1)(mynet_PECstep_tendency, pred_PEC_tendency[j,:])

    # FNO jacobian calc


    ygrad_direct_FNO[i,:,:] = torch.func.jacrev(directstep, argnums=1)(mynet_directstep_FNO, pred_direct_FNO[j,:])
    ygrad_Euler_FNO[i,:,:] = torch.func.jacrev(Eulerstep, argnums=1)(mynet_Eulerstep_FNO, pred_Euler_FNO[j,:])
    ygrad_PEC_FNO[i,:,:] = torch.func.jacrev(PECstep, argnums=1)(mynet_PECstep_FNO, pred_PEC_FNO[j,:])

    # FNO + tendency jacobian calc

    ygrad_direct_FNO_tendency[i,:,:] = torch.func.jacrev(directstep, argnums=1)(mynet_directstep_FNO_tendency, pred_direct_FNO_tendency[j,:])
    ygrad_Euler_FNO_tendency[i,:,:] = torch.func.jacrev(Eulerstep, argnums=1)(mynet_Eulerstep_FNO_tendency, pred_Euler_FNO_tendency[j,:])
    ygrad_PEC_FNO_tendency[i,:,:] = torch.func.jacrev(PECstep, argnums=1)(mynet_PECstep_FNO_tendency, pred_PEC_FNO_tendency[j,:])

    i += 1

fig1, ax1 = plt.subplots(figsize=(10,8))
fig2, ax2 = plt.subplots(figsize=(10,8))

time_vals = [0, 10000, 50000, 100000]
for i in range(4):
    ax1.plot(np.linalg.eigvals(ygrad_direct[i,:,:]), label='Direct step at t='+ time_vals[i]) #direct step basic linear
    ax1.plot(np.linalg.eigvals(ygrad_Euler[i,:,:]), label='Euler step at t='+ time_vals[i]) #euler step basic linear
    ax1.plot(np.linalg.eigvals(ygrad_RK4[i,:,:]), label='RK4 step at t='+ time_vals[i]) #RK4 step basic linear
    ax1.plot(np.linalg.eigvals(ygrad_PEC[i,:,:]), label='PEC step at t='+ time_vals[i]) #PEC step basic linear

    ax2.plot(np.linalg.eigvals(ygrad_direct_tendency[i,:,:]), label='Direct stepat t='+ time_vals[i]) #direct step tendency linear
    ax2.plot(np.linalg.eigvals(ygrad_Euler_tendency[i,:,:]), label='Euler step at t='+ time_vals[i]) #euler step tendency linear
    ax2.plot(np.linalg.eigvals(ygrad_RK4_tendency[i,:,:]), label='RK4 step at t='+ time_vals[i]) #RK4 step tendency linear
    ax2.plot(np.linalg.eigvals(ygrad_PEC_tendency[i,:,:]), label='PEC step at t='+ time_vals[i]) #PEC step tendency linear

    ax1.plot(np.linalg.eigvals(ygrad_direct_FNO[i,:,:]), label='Direct step FNO at t='+ time_vals[i]) #direct step FNO
    ax1.plot(np.linalg.eigvals(ygrad_Euler_FNO[i,:,:]), label='Euler step FNO at t='+ time_vals[i]) #euler step FNO
    ax1.plot(np.linalg.eigvals(ygrad_PEC_FNO[i,:,:]), label='PEC step FNO at t='+ time_vals[i]) #PEC step FNO

    ax2.plot(np.linalg.eigvals(ygrad_direct_FNO_tendency[i,:,:]), label='Direct step at t='+ time_vals[i]) #direct step tendency FNO
    ax2.plot(np.linalg.eigvals(ygrad_Euler_FNO_tendency[i,:,:]), label='Euler step at t='+ time_vals[i]) #euler step tendency FNO
    ax2.plot(np.linalg.eigvals(ygrad_PEC_FNO_tendency[i,:,:]), label='PEC step at t='+ time_vals[i]) #PEC step tendency FNO

fig1.savefig(path_outputs+'Eig_vals_linear.png') 
fig2.savefig(path_outputs+'Eig_vals_FNO.png')
