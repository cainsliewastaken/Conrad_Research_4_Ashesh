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
# from nn_spectral_loss import spectral_loss

lead=1

step_func = Directstep

net_file_name = 'NN_Directstep_lead'+str(lead)+'_tendeny.pt'

path_outputs = '/media/volume/sdb/conrad_stability/model_eval_tendency/'



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



mynet = MLP_Net(input_size, hidden_layer_size, output_size).cuda()
count_parameters(mynet)


#use two optimizers.  learing rates seem to work.
optimizer = optim.SGD(mynet.parameters(), lr=0.005)

loss_fn = nn.MSELoss()
epochs = 60
batch_size = 100
wavenum_init = 100
lamda_reg = 10


def spectral_loss (output, output2, target, tendency):

   loss1 = torch.mean((output-target)**2)
   
   out_fft = torch.fft.rfft(output,dim=1)
   target_fft = torch.fft.rfft(target,dim=1)
   
   loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:] - target_fft[:,wavenum_init:]))


   out_du_fft =torch.fft.rfft((output-output2)/time_step,dim=1)
   target_du_fft =torch.fft.rfft(tendency/time_step,dim=1)

   loss3 = torch.mean(torch.abs(out_du_fft[:,0:]-target_du_fft[:,0:]))

   loss = loss1 + lamda_reg*loss3
  
   return loss



for ep in range(0, epochs+1):
    for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch, du_label_batch = input_train_torch[indices], label_train_torch[indices], du_label_torch[indices]
        #pick a random boundary batch
        optimizer.zero_grad()
        outputs = step_func(mynet, input_batch, time_step)
        outputs_2 = step_func(mynet, outputs, time_step)

        # loss = loss_fn(outputs, label_batch)

        # loss = spectral_loss(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)
        loss = spectral_loss(outputs, outputs_2, label_batch, du_label_batch)

        loss.backward(retain_graph=True)
        optimizer.step()
    if ep % 10 == 0:
        print('Epoch', ep)
        print ('Loss', loss)

torch.save(mynet.state_dict(), net_file_name)
