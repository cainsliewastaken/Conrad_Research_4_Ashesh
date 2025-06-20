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
from nn_MLP import MLP_Net, MLP_net_variable
from nn_FNO import FNO1d
from nn_AE import *
from nn_spectral_loss import spectral_loss
from nn_step_methods import *
from nn_jacobian_loss import *

lead = 0
alpha = 1.0
beta = 1.0

net_name = 'MLP_AE_unit_ball_w_kde_loss_w_origin'
print(net_name)
print(alpha, beta)

path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'

# net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_implicit_lead100_spectral_jacobian_loss/chkpt_FNO_Eulerstep_implicit_lead100_spectral_jacobian_loss_epoch3.pt"
starting_epoch = 0
print('Starting epoch: ', starting_epoch)
#comment and uncomment code in training for loop below to change from mse to spectral loss in tendency

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
latent_size = 10

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()
du_label_torch = input_train_torch - label_train_torch

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

use_drop = False
mynet_encoder = MLP_net_variable(input_size, latent_size, hidden_layer_size, 2, use_act=False, use_dropout=use_drop)
mynet_decoder = MLP_net_variable(latent_size, output_size, hidden_layer_size, 4, use_act=False, use_dropout=use_drop)
mynet = AE_Net(mynet_encoder, mynet_decoder)

# mynet.load_state_dict(torch.load(net_file_path))
mynet.cuda()
count_parameters(mynet)

learning_rate = 0.0001
lr_decay = 0.975
optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

loss_fn = nn.MSELoss()
epochs = 200
batch_size = 100

torch.set_printoptions(precision=10)
# torch.autograd.set_detect_anomaly(True)
for ep in range(starting_epoch, epochs+1):
  running_loss = 0
  indices = np.random.permutation(torch.arange(trainN))
  for step in range(0,trainN,batch_size):
    batch_indices = indices[step:step + batch_size]
    input_batch, label_batch = input_train_torch[batch_indices], label_train_torch[batch_indices]

    #Use for FNO only
    input_batch = torch.reshape(input_batch,(batch_size,input_size)).float()
    label_batch = torch.reshape(label_batch,(batch_size,input_size)).float()
    input_noise = 1.0 * torch.randn_like(input_batch)

    optimizer.zero_grad()
    outputs, latent_state = mynet(input_batch)
    latent_state_noised = mynet.encode(input_noise)

    loss1 = loss_fn(outputs, label_batch)
    loss2 = mynet.KDE_uniform_loss(latent_state, alpha, beta)
    loss =  loss1 + loss2 + mynet.origin_loss(latent_state_noised) 
    loss.backward()

    optimizer.step()
    running_loss += loss.clone().detach()
  scheduler.step()

  if ep % 10 == 0:
      print('Epoch', ep)
      print ('Train Loss', float(running_loss/int(trainN/batch_size)))
      with torch.no_grad(): 
          key = np.random.randint(0, trainN, 100)
          net_out = mynet(input_train_torch[key].reshape(100,input_size).cuda().float())
          temp_loss = F.mse_loss(net_out[0], label_train_torch[key].reshape(100,input_size).cuda().float())
          print('One step loss:', float(temp_loss), 'ball loss: ', float(mynet.ball_loss(net_out[1])), 'KDE loss', float(mynet.KDE_uniform_loss(net_out[1], alpha, beta)))

      torch.save(mynet.state_dict(), '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')
 
torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")
