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
from nn_spectral_loss import spectral_loss
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step


time_step = 1e-1
lead = (1/1e-3)*time_step

step_func = PEC4step

net_name = 'NN_PEC4step_implicit_lead'+str(lead)+'_tendency'

path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

#comment and uncomment code in training for loop below to change from mse to spectral loss in tendency

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


lead=1
trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
hidden_layer_size_cascade = 1024
num_layers = 8
num_iters = 50

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()
du_label_torch = input_train_torch - label_train_torch

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])


def Eulerstep_implicit(net,input_batch,output_iter):
 output_1 = net(output_iter.cuda())
 return input_batch.cuda() + time_step*(output_1)

def PEC4step_implicit(net,input_batch,output_iter):
 output_1 = time_step*net(output_iter.cuda()) + input_batch.cuda()
 output_2 = input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_1))
 output_3 = input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_2))
 return input_batch.cuda() + time_step*0.5*(net(output_iter.cuda())+net(output_3))


def implicit_iterations(net,input_batch,output,num_iter):
    output=output.cuda()
    iter=0
    while(iter < num_iter):
      output1 = (PEC4step_implicit(net,input_batch.cuda(),output)).cuda()
      # print('residue inside implicit',torch.norm(output1-output))
      output = output1
      iter=iter+1 
    return output1


mynet = MLP_Net(input_size, hidden_layer_size, output_size).cuda()
count_parameters(mynet)

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
        indices = np.random.permutation(np.arange(start=step, step=1,stop=step+batch_size))
        input_batch, label_batch = input_train_torch[indices], label_train_torch[indices]
        optimizer.zero_grad()
        outputs = PEC4step(mynet, input_batch)
        outputs = implicit_iterations(mynet, input_batch.cuda(), outputs, num_iters)
        loss = spectral_loss(outputs, label_batch)
  
        loss.backward(retain_graph=True)
        optimizer.step()
        if ep % 5 == 0:
          print('Epoch', ep)
          print ('Loss', loss)
          torch.save(mynet.state_dict(), '/home/exouser/conrad_net_stability/Conrad_Research_4_Ashesh/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')


torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")