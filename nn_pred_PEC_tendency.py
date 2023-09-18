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


path_outputs = '/media/volume/sdb/RK4_analysis/KS_stuff/new_outputs/'

with open('/media/volume/sdb/RK4_analysis/KS_stuff/models/save/KS.pkl', 'rb') as f:
    data = pickle.load(f)


data=np.asarray(data[:,100000:])
lead=1
time_step = 1e-3
trainN=80000
input_size = 512
output_size = 512
hidden_layer_size = 1000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()

du_label_torch = input_train_torch - label_train_torch



input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()
label_test = np.transpose(data[:,trainN+lead:])




def spectral_loss (output, output2, target,tendency):

   loss1 = torch.mean((output-target)**2)
   
   out_fft = torch.fft.rfft(output,dim=1)
   target_fft = torch.fft.rfft(target,dim=1)
   
   loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:] - target_fft[:,wavenum_init:]))


   out_du_fft =torch.fft.rfft((output-output2)/time_step,dim=1)
   target_du_fft =torch.fft.rfft(tendency/time_step,dim=1)

   loss3 = torch.mean(torch.abs(out_du_fft[:,0:]-target_du_fft[:,0:]))

   loss = loss1 + lamda_reg*loss3
  
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
 output_1 = time_step*net(input_batch.cuda()) + input_batch.cuda()
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
#mynet.load_state_dict(torch.load('BNN_Spectral_Loss_lambda_reg1.0_PECstep_lead1.pt'))
mynet.load_state_dict(torch.load('BNN_Spectral_Loss_with_tendencyfft_lambda_reg5_PECstep_lead1.pt'))
count_parameters(mynet)
mynet.cuda()
epochs = 60
wavenum_init = 100
lamda_reg = 5 
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
        input_batch, label_batch, du_label_batch = input_train_torch[indices], label_train_torch[indices], du_label_torch[indices]
 #       print('shape of input_batch',input_batch.shape)
        #pick a random boundary batch
        optimizer.zero_grad()
        outputs = PEC4step(mynet,input_batch)
        outputs2 = PEC4step(mynet,outputs)
        loss = spectral_loss(outputs,outputs2,label_batch,du_label_batch)
  
        loss.backward(retain_graph=True)
        optimizer.step()
   #     epoch_loss = epoch_loss + loss
        if ep % 10 == 0:
          print('step',step)
          print('Epoch', ep)
          print ('Loss', loss)

torch.save(mynet.state_dict(),'BNN_Spectral_Loss_with_tendencyfft_'+'lambda_reg'+str(lamda_reg)+'_PEC4step_lead'+str(lead)+'.pt') 


M=20000
pred = np.zeros([M,np.size(label_test,1)])
for k in range(0,M):
 
    if (k==0):

        out = PEC4step(mynet,input_test_torch[0,:])
        pred [k,:] = out.detach().cpu().numpy()

    else:

        out = PEC4step(mynet,torch.from_numpy(pred[k-1,:]).float().cuda())

        pred [k,:] = out.detach().cpu().numpy()

matfiledata = {}
matfiledata[u'prediction'] = pred
matfiledata[u'Truth'] = label_test 
hdf5storage.write(matfiledata, '.', path_outputs+'predicted_KS_Spectral_Loss_with_tendencyfft_PEC4step_'+'lambda_reg_'+str(lamda_reg)+'_lead'+str(lead)+'.mat', matlab_compatible=True)

print('Saved Predictions')

        
