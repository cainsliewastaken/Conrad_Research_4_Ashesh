import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
#from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage
import pickle


path_outputs = '/media/volume/sdb/conrad_stability/model_eval/'

with open('/media/volume/sdb/conrad_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

lead=1
time_step = 1e-3
trainN=150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])




def spectral_loss (output, target):

   loss1 = torch.mean((output-target)**2)

   out_fft = torch.fft.rfft(output,dim=1)
   target_fft = torch.fft.rfft(target,dim=1)

   loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:] - target_fft[:,wavenum_init:]))

   loss = (1-lamda_reg)*loss1 + lamda_reg*loss2

   return loss

def RK4step(net,input_batch):
 output_1 = net(input_batch.cuda())
 output_2 = net(input_batch.cuda()+0.5*output_1)
 output_3 = net(input_batch.cuda()+0.5*output_2)
 output_4 = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1 = net(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 


def directstep(net,input_batch):
  output_1 = net(input_batch.cuda())
  return output_1

def PECstep(net,input_batch):
 output_1 = net(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))

def PEC4step(net,input_batch):
 output_1 = time_step*net(input_batch.cuda()) + input_batch.cuda()
 output_2 = input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1))
 output_3 = input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_2))
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_3))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il  = ((nn.Linear(input_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.il.weight)

        self.hidden1  = ((nn.Linear(hidden_layer_size,hidden_layer_size)))
        torch.nn.init.xavier_uniform_(self.hidden1.weight)

        self.hidden2  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden2.weight)

        self.hidden3  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden3.weight)

        self.hidden4  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden4.weight)

        self.hidden5  = (nn.Linear(hidden_layer_size,hidden_layer_size))
        torch.nn.init.xavier_uniform_(self.hidden5.weight)        

        self.ol  = nn.Linear(hidden_layer_size,output_size)
        torch.nn.init.xavier_uniform_(self.ol.weight)

        self.tanh = nn.Tanh()


    def forward(self,x):

        x1 = self.tanh(self.il(x))
        x2 = self.tanh(self.hidden1(x1))
        x3 = self.tanh(self.hidden2(x2))
        x4 = self.tanh(self.hidden3(x3))
        x5 = self.tanh(self.hidden4(x4))
        x6 = self.tanh(self.hidden5(x5))
        out =self.ol(x6)
        return out


mynet = Net()
count_parameters(mynet)
mynet.cuda()
epochs = 60
wavenum_init = 100
lamda_reg = 0 
loss_fn = nn.MSELoss()
#use two optimizers.  learing rates seem to work.
optimizer = optim.SGD(mynet.parameters(), lr=0.005)

loss_fn = nn.MSELoss()
batch_size=100

for ep in range(0, epochs+1):
#      permutation = torch.randperm(M*N)
#      epoch_loss=0
      for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1,stop=step+batch_size))
        input_batch, label_batch = input_train_torch[indices], label_train_torch[indices]
 #       print('shape of input_batch',input_batch.shape)
        #pick a random boundary batch
        optimizer.zero_grad()
        outputs = directstep(mynet,input_batch)
        loss = loss_fn(outputs,label_batch)

        loss.backward(retain_graph=True)
        optimizer.step()
   #     epoch_loss = epoch_loss + loss
        if ep % 10 == 0:
          print('step',step)
          print('Epoch', ep)
          print ('Loss', loss)

torch.save(mynet.state_dict(),'NN_Directstep_lead'+str(lead)+'.pt') 

