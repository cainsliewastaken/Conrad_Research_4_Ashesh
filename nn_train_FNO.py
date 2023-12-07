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

net_name = 'NN_FNO_Directstep_lead'+str(lead)+'_tendency'

# to changfe from normal loss to spectral loss scroll down 2 right above train for loop

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
time_step = 1e-3
trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).cuda()
du_label_torch = (input_train_torch - label_train_torch).cuda()


time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 512  # input and output chasnnels to the FNO layer

learning_rate = 0.001
lr_decay = 0.4


mynet = FNO1d(modes, width, time_future, time_history).cuda()
count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)



epochs = 60
batch_size = 50
wavenum_init = 100
lamda_reg = 5

# loss_fn = nn.MSELoss()  #for basic loss func
loss_fc = spectral_loss #for spectral loss in tendency, also change loss code inside for loop below
torch.set_printoptions(precision=10)

for ep in range(0, epochs+1):
    for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1,stop=step+batch_size))
        input_batch, label_batch, du_label_batch = input_train_torch[indices].cuda(), label_train_torch[indices].cuda(), du_label_torch[indices].cuda()
        input_batch = torch.reshape(input_batch,(batch_size,input_size,1)).float()
        label_batch = torch.reshape(label_batch,(batch_size,input_size,1)).float()
        du_label_batch = torch.reshape(du_label_batch,(batch_size,input_size,1)).float()
        #pick a random boundary batch
        
        
        optimizer.zero_grad()
        outputs = step_func(mynet, input_batch, time_step)
        
        # loss = loss_fn(outputs, label_batch)  # use this loss function for mse loss

        outputs_2 = step_func(mynet, outputs, time_step) #use this line and line below for spectral loss
        loss = loss_fc(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        loss.backward(retain_graph=True)
        

        optimizer.step()


    if ep % 5 == 0:
        print('Epoch', ep)
        print ('Loss', loss)
        # torch.save(mynet.state_dict(), '/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')


torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")