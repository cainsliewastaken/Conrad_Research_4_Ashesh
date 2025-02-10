import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_MLP import MLP_Net
from nn_step_methods import *
from nn_spectral_loss import spectral_loss
from nn_jacobian_loss import *

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead)

# step_func = PEC4step
# print(step_func)
starting_epoch = 0

net_name = 'MLP_RK4step_lead'+str(lead)+'_spectral_jac_loss'
print(net_name)
path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

# net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/MLP_PEC4step_lead1_test_loss/chkpt_MLP_PEC4step_lead1_test_loss_epoch23.pt"
# print(net_file_path)

#comment and uncomment code in training for loop below to change from mse to spectral loss in tendency

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
device = 'cuda'

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()
label_2_train_torch = torch.from_numpy(np.transpose(data[:,lead+1:lead+trainN+1])).float().cuda()

# du_label_torch = input_train_torch - label_train_torch

mynet = MLP_Net(input_size, hidden_layer_size, output_size).cuda()

# mynet.load_state_dict(torch.load(net_file_path))
# print('Model loaded')

count_parameters(mynet)

#use two optimizers.  learing rates seem to work.
# optimizer = optim.SGD(mynet.parameters(), lr=0.005)

learning_rate = 1e-4
lr_decay = 0.4
optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

loss_fn = nn.MSELoss()
epochs = 60
batch_size = 100
wavenum_init = 100
lamda_reg = 5

torch.set_printoptions(precision=10)

step_net = RK4_step(mynet, device, time_step)

print(step_net)

for ep in range(starting_epoch, epochs+1):
    running_loss = 0
    indices = np.random.permutation(torch.arange(trainN))
    for step in range(0,trainN,batch_size):
        batch_indices = indices[step:step + batch_size]
        # indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch,  = input_train_torch[batch_indices], label_train_torch[batch_indices]
        # label_batch_2 =  label_2_train_torch[batch_indices]
        #pick a random boundary batch
        optimizer.zero_grad()

        # outputs = step_func(mynet, input_batch, time_step)
        
        # loss = loss_fn(outputs, label_batch) #use this for basic mse loss 

        # outputs_2 = step_func(mynet, outputs, time_step) #use these two lines for spectral loss in tendency
        # loss = spectral_loss(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        # loss = test_loss(step_net, input_batch, label_batch, label_batch_2, 1)

        # loss = jacobian_loss(step_net, input_batch, label_batch)
        loss = spectral_jacobian_loss(step_net, input_batch, label_batch, 0, 1)
        # print(float(loss))

        loss.backward()
        optimizer.step()
        running_loss += loss.clone().detach()

    # scheduler.step()

    if ep % 1 == 0:
        print('Epoch', ep)
        print ('Train Loss', float(running_loss/int(trainN/batch_size)))
        with torch.no_grad():
            key = np.random.randint(0, trainN, 10000)
            temp_loss = F.mse_loss(step_net(input_train_torch[key]), label_train_torch[key])
            print('One step loss:', float(temp_loss))

        torch.save(mynet.state_dict(), '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')
 
torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")
