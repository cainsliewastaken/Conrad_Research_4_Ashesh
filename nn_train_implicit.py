import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
#from prettytable import PrettyTable
from count_trainable_params import count_parameters
import pickle
from nn_MLP import MLP_Net
from nn_FNO import FNO1d
from nn_spectral_loss import spectral_loss
from nn_step_methods import *
from nn_jacobian_loss import *

time_step = 1e-1
lead = int((1/1e-3)*time_step)

print(time_step, lead)

step_network = Implicit_Euler_step

net_name = 'FNO_Eulerstep_implicit_lead'+str(lead)+'_spectral_jacobian_loss'
print(net_name)

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'

net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_implicit_lead100_spectral_jacobian_loss/chkpt_FNO_Eulerstep_implicit_lead100_spectral_jacobian_loss_epoch3.pt"
starting_epoch = 4
print('Starting epoch: ', starting_epoch)
#comment and uncomment code in training for loop below to change from mse to spectral loss in tendency

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
hidden_layer_size_cascade = 1024
num_layers = 8

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()
du_label_torch = input_train_torch - label_train_torch

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


# def Eulerstep_implicit(net,input_batch,output_iter):
#  output_1 = net(output_iter.cuda())
#  return input_batch.cuda() + time_step*(output_1)

# def PEC4step_implicit(net,input_batch,output_iter):
#  output_1 = time_step*net(output_iter.cuda()) + input_batch.cuda()
#  output_2 = input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_1))
#  output_3 = input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_2))
#  return input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_3))

# def implicit_iterations(net,input_batch,output,num_iter):
#     output=output.cuda()
#     iter=0
#     while(iter < num_iter):
#       output1 = (PEC4step_implicit(net,input_batch.cuda(),output)).cuda()
#       # print('residue inside implicit',torch.norm(output1-output), iter)
#       output = output1
#       iter=iter+1 
#     return output1

# def implicit_iterations_euler(net,input_batch,output,num_iter):
#     output=output.cuda()
#     iter=0
#     while(iter < num_iter):
#       output1 = (Eulerstep_implicit(net,input_batch.cuda(),output)).cuda()
#       # print('residue inside implicit',torch.norm(output1-output), iter)
#       output = output1
#       iter=iter+1 
#     return output1

# wavenum_init = 50

# def spectral_loss_no_tendency(output, target):

#    loss1 = torch.mean((output-target)**2)
   
#    out_fft = torch.fft.rfft(output,dim=1)
#    target_fft = torch.fft.rfft(target,dim=1)
   
#    loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:] - target_fft[:,wavenum_init:]))

#    loss = (1-lamda_reg)*loss1 + lamda_reg*loss2
  
#    return loss

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 256  # input and output chasnnels to the FNO layer

# mynet = MLP_Net(input_size, hidden_layer_size, output_size).cuda()
mynet = FNO1d(modes, width, time_future, time_history)
mynet.load_state_dict(torch.load(net_file_path))
mynet.cuda()
count_parameters(mynet)

learning_rate = 0.0001
lr_decay = 0.4
optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)

loss_fn = nn.MSELoss()
epochs = 60
batch_size = 1
lamda_reg = 5

torch.set_printoptions(precision=10)

num_iters = 10
step_net = step_network(mynet, device, num_iters, time_step)

for ep in range(starting_epoch, epochs+1):
  running_loss = 0
  indices = np.random.permutation(torch.arange(trainN))
  for step in range(0,trainN,batch_size):
    batch_indices = indices[step:step + batch_size]
    input_batch, label_batch = input_train_torch[batch_indices], label_train_torch[batch_indices]

    #Use for FNO only
    input_batch = torch.reshape(input_batch,(batch_size,input_size,1)).float()
    label_batch = torch.reshape(label_batch,(batch_size,input_size,1)).float()

    optimizer.zero_grad()
    # outputs = Eulerstep(mynet, input_batch, time_step)
    # outputs = implicit_iterations_euler(mynet, input_batch.cuda(), outputs, num_iters)

    # outputs = PEC4step(mynet, input_batch, time_step)
    # outputs = implicit_iterations(mynet, input_batch.cuda(), outputs, num_iters)

    # loss = spectral_loss_no_tendency(outputs, label_batch)

    # outputs = step_net(input_batch)
    # loss = loss_fn(outputs, label_batch)

    loss = spectral_jacobian_loss(step_net, input_batch, label_batch, 1, 1)

    loss.backward(retain_graph=True)
    optimizer.step()
    running_loss += loss.clone().detach()

  if ep % 1 == 0:
      print('Epoch', ep)
      print ('Train Loss', float(running_loss/int(trainN/batch_size)))
      with torch.no_grad():
          key = np.random.randint(0, trainN, 100)
          temp_loss = F.mse_loss(step_net(input_train_torch[key].reshape(100,input_size,1).cuda().float()), label_train_torch[key].reshape(100,input_size,1).cuda().float())
          print('One step loss:', float(temp_loss))

      torch.save(mynet.state_dict(), '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')
 
torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")
