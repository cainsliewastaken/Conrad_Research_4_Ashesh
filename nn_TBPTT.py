import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_LTC_wrapper import LTC_Concat_net
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss

lead=1

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_LTC_tendency/'

step_func = Directstep

net_file_name = 'NN_LTC_Directstep_lead'+str(lead)+'_tendency.pt'



with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
num_hidden_neurons = 6


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


mynet = LTC_Concat_net(input_size, num_hidden_neurons).cuda()
count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)



epochs = 60
wavenum_init = 100
lamda_reg = 5

# loss_fn = nn.MSELoss()
loss_fc = spectral_loss
torch.set_printoptions(precision=10)

state_buffer_len = 20
backprop_every_n = 10
for ep in range(0, epochs+1):
    #init starting hidden state
    init_state =  mynet.init_hidden_state #make this currently a vector of trainable params
    states = [(None, init_state)]
    for current_step in range(0,trainN):
        input_train_torch, label_train_torch, du_label_torch

        state = states[-1][1].detach() 
        state.requires_grad=True
        #detach the last hidden state from all previous hidden states

        outputs = step_func(mynet, input_train_torch[current_step], time_step)
        x_out_true = outputs[0:mynet.true_x_size-1]
        new_state = outputs[mynet.true_x_size:]

        states.append((state, new_state))
        while len(states) > state_buffer_len:
            del states[0]
        #trim the stored previous states

        if (current_step+1)%backprop_every_n==0:
            # loss = loss_fn(x_out_true, label_train_torch[current_step])

            outputs_2 = step_func(mynet, outputs, time_step)
            x_out_true_2 = outputs_2[0:mynet.true_x_size-1]
            loss = loss_fc(x_out_true, x_out_true_2, label_train_torch[current_step], du_label_torch[current_step], wavenum_init, lamda_reg, time_step)

            optimizer.zero_grad()

            for i in range(0, state_buffer_len-1):
                if states[-i-1][0] is None:
                    break
                    #this stops us from backproping past the init state
                current_grad = states[-i-1][0].grad
                states[-i-2][1].backward(current_grad, retain_graph=True)
        optimizer.step()


    if ep % 5 == 0:
        print('Epoch', ep)
        print ('Loss', loss)

torch.save(mynet.state_dict(), net_file_name)
torch.set_printoptions(precision=4)