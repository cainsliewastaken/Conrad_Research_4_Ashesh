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
import matplotlib.animation as anim

from nn_FNO import FNO1d
from nn_MLP import MLP_Net 
# from nn_step_methods import Directstep, Eulerstep, PECstep

from torch.profiler import profile, record_function, ProfilerActivity
import gc


lead = 1
path_outputs = '/media/volume/sdb/conrad_stability/jacobian_mats_all_models/'

model_path = "/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/NN_FNO_Directstep_lead1_tendency.pt"

matfile_name = 'FNO_KS_PECstep_tendency_lead'+str(lead)+'_UNTRAINED_jacs.mat'


print('loading data')


with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

print('data loaded')


lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
hidden_layer_size = 2000
output_size = 1024



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float()
label_test = np.transpose(data[:,trainN+lead:])


eq_points = 4
# FNO archetecture hyperparams

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 1500 # input and output chasnnels to the FNO layer




x_torch = torch.zeros([eq_points,input_size]).cuda()

count=0
for k in (np.array([ int(0),  int(10000), int(20000), int(99999)])):
  x_torch[count,:] = input_test_torch[k,:].requires_grad_(requires_grad=True)
  count=count+1



# mynet = MLP_Net(input_size, hidden_layer_size, output_size)
mynet = FNO1d(modes, width, time_future, time_history)
# mynet.load_state_dict(torch.load(model_path))
print('model defined')
print(torch.cuda.memory_allocated())
mynet.cuda()
print('model cuda')
print(torch.cuda.memory_allocated())

mynet.eval()


def RK4step(input_batch):
 output_1 = mynet(input_batch.cuda())
 output_2 = mynet(input_batch.cuda()+0.5*output_1)
 output_3 = mynet(input_batch.cuda()+0.5*output_2)
 output_4 = mynet(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6

def Eulerstep(input_batch):
 output_1 = mynet(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 
  
def Directstep(input_batch):
  output_1 = mynet(input_batch.cuda())
  return output_1

def PECstep(input_batch):
 output_1 = mynet(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(mynet(input_batch.cuda())+mynet(output_1))

# print(torch.cuda.memory_allocated())
step_func = Directstep

# print(torch.cuda.memory_allocated())

ygrad = torch.zeros([eq_points,input_size,input_size])



# record_shapes=True, profile_memory=True,

with profile(activities=[ProfilerActivity.CPU ,ProfilerActivity.CUDA],  with_stack=True) as prof:
  mynet(torch.reshape(input_test_torch[0,:].cuda(),(1,input_size,1)))

# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_memory_usage", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))



for k in range(0,eq_points):
  # for obj in gc.get_objects():
  #   try:
  #       if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
  #           print(type(obj), obj.size())
  #   except:
  #       pass
    # ygrad [k,:,:] = torch.autograd.functional.jacobian(step_func,x_torch[k,:]) #Use these 2 lines for MLP networks
    
    temp_mat = torch.autograd.functional.jacobian(step_func, torch.reshape(x_torch[k,:],(1,input_size,1))) #Use these for FNO
    ygrad [k,:,:] = torch.reshape(temp_mat,(1,input_size, input_size))

    # print(sum(sum(np.abs(ygrad[k,:,:]))))



ygrad = ygrad.detach().cpu().numpy()

print(ygrad.shape)



matfiledata = {}
matfiledata[u'Jacobian_mats'] = ygrad
scipy.io.savemat(path_outputs+matfile_name, matfiledata)

print('Saved Predictions')



-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                            aten::copy_         0.11%     410.000us         0.77%       2.870ms     114.800us     202.888ms        93.67%     202.890ms       8.116ms            25
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     202.811ms        93.64%     202.811ms       2.669ms            76
                                              aten::bmm         0.14%     533.000us        60.83%     225.509ms      45.102ms      12.943ms         5.98%     215.695ms      43.139ms             5
std::enable_if<!(false), void>::type internal::gemvx...         0.00%       0.000us         0.00%       0.000us       0.000us      12.905ms         5.96%      12.905ms       3.226ms             4
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     244.000us         0.11%     244.000us      40.667us             6
                                aten::cudnn_convolution        20.57%      76.241ms        26.23%      97.227ms      24.307ms     197.000us         0.09%     197.000us      49.250us             4
void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_1...         0.00%       0.000us         0.00%       0.000us       0.000us     193.000us         0.09%     193.000us      48.250us             4
                                            aten::fill_         0.02%      78.000us         1.16%       4.291ms     536.375us      59.000us         0.03%      59.000us       7.375us             8
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      59.000us         0.03%      59.000us       7.375us             8
                         Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      52.000us         0.02%      52.000us      13.000us             4
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 370.729ms
Self CUDA time total: 216.588ms


