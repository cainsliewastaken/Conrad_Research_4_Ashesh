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
mynet_directstep = Net().double()
mynet_directstep.load_state_dict(torch.load('NN_directstep_lead1.pt'))
mynet_directstep.cuda()

mynet_Eulerstep = Net().double()
mynet_Eulerstep.load_state_dict(torch.load('NN_Eulerstep_lead1.pt'))
mynet_Eulerstep.cuda()

mynet_RK4step = Net().double()
mynet_RK4step.load_state_dict(torch.load('NN_RK4step_lead1.pt'))
mynet_RK4step.cuda()

mynet_PECstep = Net().double()
mynet_RK4step.load_state_dict(torch.load('NN_PECstep_lead1.pt'))
mynet_PECstep.cuda()


val_dict_direct = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval/predicted_directstep_1024_lead1.mat')
val_dict_Euler = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval/predicted_Eulerstep_1024_lead1.mat')
val_dict_RK4 = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval/predicted_RK4step_1024_lead1.mat')
val_dict_PEC = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval/predicted_PECstep_1024_lead1.mat')

ks_truth = val_dict_direct[u'Truth']


pred_direct = val_dict_direct[u'prediction']
pred_Euler = val_dict_Euler[u'prediction']
pred_RK4 = val_dict_RK4[u'prediction']
pred_PEC = val_dict_PEC[u'prediction']


ygrad_direct = torch.zeros([int(4),input_size,input_size])
ygrad_Euler = torch.zeros([int(4),input_size,input_size])
ygrad_RK4 = torch.zeros([int(4),input_size,input_size])
ygrad_PEC = torch.zeros([int(4),input_size,input_size])



#tendency models and predictions
mynet_directstep_tendency = Net().double()
mynet_directstep_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_directstep_lead1.pt'))
mynet_directstep_tendency.cuda()

mynet_Eulerstep_tendency = Net().double()
mynet_Eulerstep_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_Eulerstep_lead1.pt'))
mynet_Eulerstep_tendency.cuda()

mynet_RK4step_tendency = Net().double()
mynet_RK4step_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_RK4step_lead1.pt'))
mynet_RK4step_tendency.cuda()

mynet_PECstep_tendency = Net().double()
mynet_RK4step_tendency.load_state_dict(torch.load('NN_Spectral_Loss_with_tendencyfft_lambda_reg5_PECstep_lead1.pt'))
mynet_PECstep_tendency.cuda()

val_dict_direct_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_tendency/predicted_directstep_1024_lead1_lambda_reg5_tendency.mat')
val_dict_Euler_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_tendency/predicted_Eulerstep_1024_lead1_lambda_reg5_tendency.mat')
val_dict_RK4_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_tendency/predicted_RK4step_1024_lead1_lambda_reg5_tendency.mat')
val_dict_PEC_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_tendency/predicted_PECstep_1024_lead1_lambda_reg5_tendency.mat')

pred_direct_tendency = val_dict_direct_tendency[u'prediction']
pred_Euler_tendency = val_dict_Euler_tendency[u'prediction']
pred_RK4_tendency = val_dict_RK4_tendency[u'prediction']
pred_PEC_tendency = val_dict_PEC_tendency[u'prediction']

ygrad_direct_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_RK4_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_tendency = torch.zeros([int(4),input_size,input_size])

print('Linear is loaded')


i = 0
for j in np.array([0, 10000, 50000, 99998]):
    #basic linear model jacobian calculation

    ygrad_direct[i,:,:] = torch.func.jacfwd(directstep, argnums=1)(mynet_directstep, torch.tensor(pred_direct[j,:], dtype=torch.double))
    ygrad_Euler[i,:,:] = torch.func.jacfwd(Eulerstep, argnums=1)(mynet_Eulerstep, torch.tensor(pred_Euler[j,:], dtype=torch.double))
    ygrad_RK4[i,:,:] = torch.func.jacfwd(RK4step, argnums=1)(mynet_RK4step, torch.tensor(pred_RK4[j,:], dtype=torch.double))
    ygrad_PEC[i,:,:] = torch.func.jacfwd(PECstep, argnums=1)(mynet_PECstep, torch.tensor(pred_PEC[j,:], dtype=torch.double))

    #linear plus tendency (spectral loss) jacobian calculation

    ygrad_direct_tendency[i,:,:] = torch.func.jacfwd(directstep, argnums=1)(mynet_directstep_tendency, torch.tensor(pred_direct_tendency[j,:], dtype=torch.double))
    ygrad_Euler_tendency[i,:,:] = torch.func.jacfwd(Eulerstep, argnums=1)(mynet_Eulerstep_tendency, torch.tensor(pred_Euler_tendency[j,:], dtype=torch.double))
    ygrad_RK4_tendency[i,:,:] = torch.func.jacfwd(RK4step, argnums=1)(mynet_RK4step_tendency, torch.tensor(pred_RK4_tendency[j,:], dtype=torch.double))
    ygrad_PEC_tendency[i,:,:] = torch.func.jacfwd(PECstep, argnums=1)(mynet_PECstep_tendency, torch.tensor(pred_PEC_tendency[j,:], dtype=torch.double))
    i += 1
print('Linear jacs calculated')

torch.cuda.empty_cache()

del mynet_directstep, mynet_directstep_tendency, mynet_Eulerstep, mynet_Eulerstep_tendency
del mynet_PECstep, mynet_PECstep_tendency, mynet_RK4step, mynet_RK4step_tendency



# FNO archetecture hyperparams

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 64 # input and output chasnnels to the FNO layer

num_epochs = 1 #set to one so faster computation, in principle 20 is best.  WHERE IS THIS USED, WHAT IT DO?
learning_rate = 0.0001
lr_decay = 0.4
num_workers = 0  #What does this do?




# FNO models and predictions
mynet_directstep_FNO = FNO1d(modes, width, time_future, time_history).float()
mynet_directstep_FNO.load_state_dict(torch.load('NN_FNO_Directstep_lead1.pt'))
mynet_directstep_FNO.cuda()

mynet_Eulerstep_FNO = FNO1d(modes, width, time_future, time_history).float()
mynet_Eulerstep_FNO.load_state_dict(torch.load('NN_FNO_Eulerstep_lead1.pt'))
mynet_Eulerstep_FNO.cuda()

mynet_PECstep_FNO = FNO1d(modes, width, time_future, time_history).float()
mynet_PECstep_FNO.load_state_dict(torch.load('NN_FNO_PECstep_lead1.pt'))
mynet_PECstep_FNO.cuda()

val_dict_direct_FNO = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_directstep_1024_FNO_lead1.mat')
val_dict_Euler_FNO = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_Eulerstep_1024_FNO_lead1.mat')
val_dict_PEC_FNO = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO/predicted_PECstep_1024_FNO_lead1.mat')

pred_direct_FNO = val_dict_direct_FNO[u'prediction']
pred_Euler_FNO = val_dict_Euler_FNO[u'prediction']
pred_PEC_FNO = val_dict_PEC_FNO[u'prediction']

ygrad_direct_FNO = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_FNO = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_FNO = torch.zeros([int(4),input_size,input_size])



i = 0
for j in np.array([0, 10000, 50000, 99998]):
    # FNO jacobian calc

    ygrad_direct_FNO[i,:,:] = torch.func.jacfwd(directstep, argnums=1)(mynet_directstep_FNO, torch.reshape(torch.tensor(pred_direct_FNO[0,:], dtype=torch.float),(1,input_size,1)))
    ygrad_Euler_FNO[i,:,:] = torch.func.jacfwd(Eulerstep, argnums=1)(mynet_Eulerstep_FNO, torch.reshape(torch.tensor(pred_Euler_FNO[0,:], dtype=torch.float),(1,input_size,1)))
    ygrad_PEC_FNO[i,:,:] = torch.func.jacfwd(PECstep, argnums=1)(mynet_PECstep_FNO, torch.reshape(torch.tensor(pred_PEC_FNO[0,:], dtype=torch.float),(1,input_size,1)))
    i += 1


print('FNO basic jacs calculated')
torch.cuda.empty_cache()

del mynet_directstep_FNO, mynet_Eulerstep_FNO, mynet_PECstep_FNO


# FNO + tendency models and predictions
mynet_directstep_FNO_tendency = FNO1d(modes, width, time_future, time_history).float()
mynet_directstep_FNO_tendency.load_state_dict(torch.load('NN_Spectral_Loss_FNO_Directstep_tendency_lambda_reg5lead1.pt'))
mynet_directstep_FNO_tendency.cuda() 

mynet_Eulerstep_FNO_tendency = FNO1d(modes, width, time_future, time_history).float()
mynet_Eulerstep_FNO_tendency.load_state_dict(torch.load('NN_Spectral_Loss_FNO_Eulerstep_tendency_lambda_reg5lead1.pt'))
mynet_Eulerstep_FNO_tendency.cuda()

mynet_PECstep_FNO_tendency = FNO1d(modes, width, time_future, time_history).float()
mynet_PECstep_FNO_tendency.load_state_dict(torch.load('NN_Spectral_Loss_FNO_PECstep_tendency_lambda_reg5lead1.pt'))
mynet_PECstep_FNO_tendency.cuda()

val_dict_direct_FNO_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/predicted_directstep_1024_FNO_tendency_lead1.mat')
val_dict_Euler_FNO_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/predicted_Eulerstep_1024_FNO_tendency_lead1.mat')
val_dict_PEC_FNO_tendency = scipy.io.loadmat('/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/predicted_PECstep_1024_FNO_tendency_lead1.mat')

pred_direct_FNO_tendency = val_dict_direct_FNO_tendency[u'prediction']
pred_Euler_FNO_tendency = val_dict_Euler_FNO_tendency[u'prediction']
pred_PEC_FNO_tendency = val_dict_PEC_FNO_tendency[u'prediction']

ygrad_direct_FNO_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_Euler_FNO_tendency = torch.zeros([int(4),input_size,input_size])
ygrad_PEC_FNO_tendency = torch.zeros([int(4),input_size,input_size])

print('FNO is loaded')


i = 0
for j in np.array([0, 10000, 50000, 99998]):
    # FNO + tendency jacobian calc

    ygrad_direct_FNO_tendency[i,:,:] = torch.func.jacfwd(directstep, argnums=1)(mynet_directstep_FNO_tendency, torch.reshape(torch.tensor(pred_direct_FNO_tendency[0,:], dtype=torch.float),(1,input_size,1)))
    ygrad_Euler_FNO_tendency[i,:,:] = torch.func.jacfwd(Eulerstep, argnums=1)(mynet_Eulerstep_FNO_tendency, torch.reshape(torch.tensor(pred_Euler_FNO_tendency[0,:], dtype=torch.float),(1,input_size,1)))
    ygrad_PEC_FNO_tendency[i,:,:] = torch.func.jacfwd(PECstep, argnums=1)(mynet_PECstep_FNO_tendency, torch.reshape(torch.tensor(pred_PEC_FNO_tendency[0,:], dtype=torch.float),(1,input_size,1)))
    i += 1


print('All jacobians calculated')
torch.cuda.empty_cache()

fig1, ax1 = plt.subplots(figsize=(10,8))
fig2, ax2 = plt.subplots(figsize=(10,8))
fig3, ax3 = plt.subplots(figsize=(10,8))
circ = plt.Circle((0,0), radius=1)
ax1.add_patch(circ)
ax2.add_patch(circ)
time_vals = [0, 10000, 50000, 99998]
for i in range(4):
    eig_direct = np.linalg.eigvals(ygrad_direct[i,:,:])
    sing_direct = np.linalg.svd(ygrad_direct[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_direct)), label='Direct step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_direct)), label='Direct step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_direct.real, eig_direct.imag, label='Direct step at t='+ time_vals[i]) #direct step basic linear
    eig_Euler = np.linalg.eigvals(ygrad_Euler[i,:,:])
    sing_Euler = np.linalg.svd(ygrad_Euler[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_Euler)), label='Euler step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_Euler)), label='Euler step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_Euler.real, eig_Euler.imag, label='Euler step at t='+ time_vals[i]) #euler step basic linear
    eig_RK4 = np.linalg.eigvals(ygrad_RK4[i,:,:])
    sing_RK4 = np.linalg.svd(ygrad_RK4[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_RK4)), label='RK4 step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_RK4)), label='RK4 step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_RK4.real, eig_RK4.imag, label='RK4 step at t='+ time_vals[i]) #RK4 step basic linear
    eig_PEC = np.linalg.eigvals(ygrad_PEC[i,:,:])
    sing_PEC = np.linalg.svd(ygrad_PEC[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_PEC)), label='PEC step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_PEC)), label='PEC step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_PEC.real, eig_PEC.imag, label='PEC step at t='+ time_vals[i]) #PEC step basic linear
    
    eig_direct_tendency = np.linalg.eigvals(ygrad_direct_tendency[i,:,:])
    sing_direct_tendency = np.linalg.svd(ygrad_direct_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_direct_tendency)), label='Tendency Direct step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_direct_tendency)), label='Tendency Direct step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_direct_tendency.real, eig_direct_tendency.imag, label='Direct stepat t='+ time_vals[i]) #direct step tendency linear
    eig_Euler_tendency = np.linalg.eigvals(ygrad_Euler_tendency[i,:,:])
    sing_Euler_tendency = np.linalg.svd(ygrad_Euler_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_Euler_tendency)), label='Tendency Euler step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_Euler_tendency)), label='Tendency Euler step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_Euler_tendency.real, eig_Euler_tendency.imag, label='Euler step at t='+ time_vals[i]) #euler step tendency linear
    eig_RK4_tendency = np.linalg.eigvals(ygrad_RK4_tendency[i,:,:])
    sing_RK4_tendency = np.linalg.svd(ygrad_RK4_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_RK4_tendency)), label='Tendency RK4 step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_RK4_tendency)), label='Tendency RK4 step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_RK4_tendency.real, eig_RK4_tendency.imag, label='RK4 step at t='+ time_vals[i]) #RK4 step tendency linear
    eig_PEC_tendency = np.linalg.eigvals(ygrad_PEC_tendency[i,:,:])
    sing_PEC_tendency = np.linalg.svd(ygrad_PEC_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_PEC_tendency)), label='Tendency PEC step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_PEC_tendency)), label='Tendency PEC step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_PEC_tendency.real, eig_PEC_tendency.imag, label='PEC step at t='+ time_vals[i]) #PEC step tendency linear


    eig_direct_FNO = np.linalg.eigvals(ygrad_direct_FNO[i,:,:])
    sing_direct_FNO = np.linalg.svd(ygrad_direct_FNO[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_direct_FNO)), label='FNO Direct step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_direct_FNO)), label='FNO Direct step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_direct_FNO.real, eig_direct_FNO.imag, label='Direct step FNO at t='+ time_vals[i]) #direct step FNO
    eig_Euler_FNO = np.linalg.eigvals(ygrad_Euler_FNO[i,:,:])
    sing_Euler_FNO = np.linalg.svd(ygrad_Euler_FNO[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_Euler_FNO)), label='FNO Euler step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_Euler_FNO)), label='FNO Euler step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_Euler_FNO.real, eig_Euler_FNO.imag, label='Euler step FNO at t='+ time_vals[i]) #euler step FNO
    eig_PEC_FNO = np.linalg.eigvals(ygrad_PEC_FNO[i,:,:])
    sing_PEC_FNO = np.linalg.svd(ygrad_PEC_FNO[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_PEC_FNO)), label='FNO PEC step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_PEC_FNO)), label='FNO PEC step sing sum at t='+ time_vals[i])
    ax1.scatter(eig_PEC_FNO.real, eig_PEC_FNO.imag, label='PEC step FNO at t='+ time_vals[i]) #PEC step FNO


    eig_direct_FNO_tendency = np.linalg.eigvals(ygrad_direct_FNO_tendency[i,:,:])
    sing_direct_FNO_tendency = np.linalg.svd(ygrad_direct_FNO_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_direct_FNO_tendency)), label='FNO Tendency Direct step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_direct_FNO_tendency)), label='FNO Tendency Direct step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_PEC_FNO.real, eig_PEC_FNO.imag, label='Direct step at t='+ time_vals[i]) #direct step tendency FNO
    eig_Euler_FNO_tendency = np.linalg.eigvals(ygrad_Euler_FNO_tendency[i,:,:])
    sing_Euler_FNO_tendency = np.linalg.svd(ygrad_Euler_FNO_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_Euler_FNO_tendency)), label='FNO Tendency Euler step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_Euler_FNO_tendency)), label='FNO Tendency Euler step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_Euler_FNO_tendency.real, eig_Euler_FNO_tendency.imag, label='Euler step at t='+ time_vals[i]) #euler step tendency FNO
    eig_PEC_FNO_tendency = np.linalg.eigvals(ygrad_PEC_FNO_tendency[i,:,:])
    sing_PEC_FNO_tendency = np.linalg.svd(ygrad_PEC_FNO_tendency[i,:,:], compute_uv=False)
    ax3.scatter(i, np.absolute(np.sum(eig_PEC_FNO_tendency)), label='FNO Tendency PEC step eigvals sum at t='+ time_vals[i])
    ax3.scatter(i, np.sum(np.absolute(sing_PEC_FNO_tendency)), label='FNO Tendency PEC step sing sum at t='+ time_vals[i])
    ax2.scatter(eig_PEC_FNO_tendency.real, eig_PEC_FNO_tendency.imag, label='PEC step at t='+ time_vals[i]) #PEC step tendency FNO
ax1.legend(fontsize='x-small')
ax2.legend(fontsize='x-small')
ax3.legend(fontsize='x-small')

fig1.savefig('Eig_vals_linear.pdf') 
fig2.savefig('Eig_vals_FNO.pdf')
fig3.savefig('Eig_Sing_vals_sum.pdf')
print('All plots plotted')
fig4, ax4 = plt.subplots(figsize=(10,8))


def ani_func_linear(t):
    ax4.cla()
    x_vals = np.arange(np.length(ks_truth[0,:]))
    y_vals = ks_truth[t,:]
    ax4.plot(x_vals, y_vals, label='Truth')

    x_vals = np.arange(np.length(pred_direct[0,:]))
    y_vals = pred_direct[t,:]
    ax4.plot(x_vals, y_vals, label='Direct Step linear')
    x_vals = np.arange(np.length(pred_PEC[0,:]))
    y_vals = pred_PEC[t,:]
    ax4.plot(x_vals, y_vals, label='PEC Step linear')

    x_vals = np.arange(np.length(pred_direct_tendency[0,:]))
    y_vals = pred_direct_tendency[t,:]
    ax4.plot(x_vals, y_vals, label='Direct Step linear spectral loss')
    x_vals = np.arange(np.length(pred_PEC_tendency[0,:]))
    y_vals = pred_PEC_tendency[t,:]
    ax4.plot(x_vals, y_vals, label='PEC Step linear spectral loss')
    ax4.legend(fontsize='x-small')

    return ax4

animation_lin = plt.animation.FuncAnimation(fig4, func=ani_func_linear, frames=range(500))
fig5, ax5 = plt.subplots(figsize=(10,8))



def ani_func_FNO(t):
    ax5.cla()
    x_vals = np.arange(np.length(ks_truth[0,:]))
    y_vals = ks_truth[t,:]
    ax5.plot(x_vals, y_vals, label='Truth')

    x_vals = np.arange(np.length(pred_direct_FNO[0,:]))
    y_vals = pred_direct_FNO[t,:]
    ax5.plot(x_vals, y_vals, label='Direct Step linear')
    x_vals = np.arange(np.length(pred_PEC_FNO[0,:]))
    y_vals = pred_PEC_FNO[t,:]
    ax5.plot(x_vals, y_vals, label='PEC Step linear')

    x_vals = np.arange(np.length(pred_direct_FNO_tendency[0,:]))
    y_vals = pred_direct_FNO_tendency[t,:]
    ax5.plot(x_vals, y_vals, label='Direct Step linear spectral loss')
    x_vals = np.arange(np.length(pred_PEC_FNO_tendency[0,:]))
    y_vals = pred_PEC_FNO_tendency[t,:]
    ax5.plot(x_vals, y_vals, label='PEC Step linear spectral loss')
    ax5.legend(fontsize='x-small')

    return ax5

animation_FNO = plt.animation.FuncAnimation(fig5, func=ani_func_FNO, frames=range(500))

writer = plt.animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
animation_lin.save('linear_evals.gif', writer=writer)
animation_FNO.save('FNO_evals.gif', writer=writer)
print('Movies saved')