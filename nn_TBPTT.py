import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_FNO import FNO1d
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss

lead=1

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_FNO_tendency/'

step_func = Directstep

net_file_name = 'NN_FNO_Directstep_lead'+str(lead)+'_tendency.pt'



with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()
du_label_torch = input_train_torch - label_train_torch

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 512 # number of Fourier modes to multiply
width = 64 # input and output chasnnels to the FNO layer

learning_rate = 0.0001
lr_decay = 0.4


mynet = FNO1d(modes, width, time_future, time_history).cuda()
count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)



epochs = 60
batch_size = 100
wavenum_init = 100
lamda_reg = 5

# loss_fn = nn.MSELoss()
loss_fc = spectral_loss
torch.set_printoptions(precision=10)

init_state =   #make this a vector of zeros, random numbers, or trainable params

for ep in range(0, epochs+1):
    #init starting hidden state
    states = [(None, init_state)]
    for step in range(0,trainN):
        input_train_torch, label_train_torch, du_label_torch
        # input_batch = torch.reshape(input_batch,(batch_size,input_size,1))
        # label_batch = torch.reshape(label_batch,(batch_size,input_size,1))
        # du_label_batch = torch.reshape(du_label_batch,(batch_size,input_size,1))
        #create batch
        state = states[-1][1].detach() #detach the last hidden state from all previous hidden states
        state.requires_grad=True
        output, new_state = 

        loss = 0
        optimizer.zero_grad()
        for current_time_step in range(0, batch_size):
            outputs = step_func(mynet, input_train_torch[current_time_step], time_step)
            
            # loss = loss_fn(outputs, label_batch)

            outputs_2 = step_func(mynet, outputs, time_step)
            loss = loss_fc(outputs, outputs_2, label_train_torch[current_time_step], du_label_torch[current_time_step], wavenum_init, lamda_reg, time_step)
            #compute loss in relation to all elements in batch
        loss.backward(retain_graph=True)
        optimizer.step()


    if ep % 5 == 0:
        print('Epoch', ep)
        print ('Loss', loss)

torch.save(mynet.state_dict(), net_file_name)
torch.set_printoptions(precision=4)