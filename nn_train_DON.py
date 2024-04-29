import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_DON import DeepONet_FullGrid
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss
from nn_Cascade_MLP import Cascade_MLP_Net



time_step = 1e-3
lead = int((1/1e-3)*time_step)

step_func = RK4step

net_name = 'DON_RK4step_lead'+str(lead)+''

model_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/DON_PEC4step_lead1/chkpt_DON_PEC4step_lead1_epoch88.pt"


path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

#comment and uncomment code in training for loop below to change from mse to spectral loss in tendency

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f:
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


num_of_basis_funcs = 200
layer_sizes_branch = [1024, 4096, 4096, 4096, num_of_basis_funcs]
layer_sizes_trunk = [1, 521, 512, 512, num_of_basis_funcs]

mynet = DeepONet_FullGrid(layer_sizes_branch, layer_sizes_trunk, 1024).cuda()
count_parameters(mynet)



#use two optimizers.  learing rates seem to work.
# optimizer = optim.SGD(mynet.parameters(), lr=0.005)

learning_rate = 0.001
lr_decay = 0.4
optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)

loss_fn = nn.MSELoss() 
epochs = 100
batch_size = 10
wavenum_init = 100
lamda_reg = 5

torch.set_printoptions(precision=10)
best_epoch_loss = np.inf
for ep in range(0, epochs+1):
    epoch_loss = 0
    for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch, du_label_batch = input_train_torch[indices], label_train_torch[indices], du_label_torch[indices]
        #pick a random boundary batch
        optimizer.zero_grad()
        outputs = step_func(torch.func.vmap(mynet), input_batch, time_step)
        
        loss = loss_fn(outputs, label_batch) #use this for basic mse loss 
        
        # outputs_2 = step_func(torch.func.vmap(mynet), outputs, time_step) #use these two lines for spectral loss in tendency
        # loss = spectral_loss(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        loss.backward()
        optimizer.step()
        epoch_loss += loss
        if ep == 1:
            print(loss, ep)
    if epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        best_chkpt = mynet.state_dict()
        
    if ep % 1 == 0:
        print('Epoch', ep)
        print ('Loss', loss)
        torch.save(mynet.state_dict(), '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')

torch.save(mynet.state_dict(), net_name+'.pt')
torch.save(best_chkpt, net_name+'_best_chkpt.pt')
torch.set_printoptions(precision=4)
print("Model Saved")