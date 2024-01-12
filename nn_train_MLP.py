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
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss
from nn_Cascade_MLP import Cascade_MLP_Net

lead=1

step_func = Directstep

net_name = 'NN_Directstep_lead'+str(lead)+''

path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

#comment and uncomment code in training for loop below to change from mse to spectral loss in tendency

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])



lead=1
time_step = 1e-3
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



mynet = MLP_Net(input_size, hidden_layer_size, output_size).cuda()
# mynet = Cascade_MLP_Net(input_size, hidden_layer_size_cascade, output_size, num_layers).cuda()
count_parameters(mynet)


#use two optimizers.  learing rates seem to work.
# optimizer = optim.SGD(mynet.parameters(), lr=0.005)

learning_rate = 0.0001
lr_decay = 0.4
optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)



loss_fn = nn.MSELoss()
epochs = 60
batch_size = 100
wavenum_init = 100
lamda_reg = 5

torch.set_printoptions(precision=10)

for ep in range(0, epochs+1):
    for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch, du_label_batch = input_train_torch[indices], label_train_torch[indices], du_label_torch[indices]
        #pick a random boundary batch
        optimizer.zero_grad()
        outputs = step_func(mynet, input_batch, time_step)
        
        # loss = loss_fn(outputs, label_batch) #use this for basic mse loss 

        
        outputs_2 = step_func(mynet, outputs, time_step) #use these two lines for spectral loss in tendency
        loss = spectral_loss(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        loss.backward(retain_graph=True)
        optimizer.step()
    if ep % 5 == 0:
        print('Epoch', ep)
        print ('Loss', loss)
        torch.save(mynet.state_dict(), '/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')


torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")