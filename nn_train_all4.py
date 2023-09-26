import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
#import netCDF4 as nc
#from prettytable import PrettyTable
#from count_trainable_params import count_parameters    
#import hdf5storage
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


if __name__ == "__main__":




  mynet_directstep = Net()
  mynet_Eulerstep = Net()
  mynet_RK4step = Net()
  mynet_PECstep = Net()


  #count_parameters(mynet_directstep)
  mynet_directstep.cuda()

  #count_parameters(mynet_Eulerstep)
  mynet_Eulerstep.cuda()

  #count_parameters(mynet_RK4step)
  mynet_RK4step.cuda()

  #count_parameters(mynet_PECstep)
  mynet_PECstep.cuda()


  epochs = 60

  optimizer_direct = optim.SGD(mynet_directstep.parameters(), lr=0.005)
  optimizer_Euler = optim.SGD(mynet_Eulerstep.parameters(), lr=0.005)
  optimizer_RK4 = optim.SGD(mynet_RK4step.parameters(), lr=0.005)
  optimizer_PEC = optim.SGD(mynet_PECstep.parameters(), lr=0.005)


  loss_fn = nn.MSELoss()
  batch_size=100

  for ep in range(0, epochs+1):
        for step in range(0,trainN,batch_size):
          indices = np.random.permutation(np.arange(start=step, step=1,stop=step+batch_size))
          input_batch, label_batch = input_train_torch[indices], label_train_torch[indices]
          #pick a random boundary batch

          #train direct_step net
          optimizer_direct.zero_grad()
          outputs_direct = directstep(mynet_directstep,input_batch)
          loss = loss_fn(outputs_direct,label_batch)
    
          loss.backward(retain_graph=True)
          optimizer_direct.step()

          #train Euler_step net
          optimizer_Euler.zero_grad()
          outputs_Euler = Eulerstep(mynet_Eulerstep,input_batch)
          loss = loss_fn(outputs_Euler,label_batch)
    
          loss.backward(retain_graph=True)
          optimizer_Euler.step()

          #train RK4_step net
          optimizer_RK4.zero_grad()
          outputs_RK4 = RK4step(mynet_RK4step,input_batch)
          loss = loss_fn(outputs_RK4,label_batch)
    
          loss.backward(retain_graph=True)
          optimizer_RK4.step()

          #train PEC_step net
          optimizer_PEC.zero_grad()
          outputs_PEC = PECstep(mynet_PECstep,input_batch)
          loss = loss_fn(outputs_PEC,label_batch)
    
          loss.backward(retain_graph=True)
          optimizer_PEC.step()

  #    #     epoch_loss = epoch_loss + loss
  #         if ep % 10 == 0:
  #           print('step',step)
  #           print('Epoch', ep)
  #           print ('Loss', loss)


  #save network
  torch.save(mynet_directstep.state_dict(),'NN_directstep_lead'+str(lead)+'.pt') 
  torch.save(mynet_Eulerstep.state_dict(),'NN_Eulerstep_lead'+str(lead)+'.pt') 
  torch.save(mynet_RK4step.state_dict(),'NN_RK4step_lead'+str(lead)+'.pt') 
  torch.save(mynet_PECstep.state_dict(),'NN_PECstep_lead'+str(lead)+'.pt') 

  print('Saved Models')


# creates and store predictions using last 100000 timesteps, this process is now included in eval_networks_plot_rlts.py

# M=100000
# pred_direct = np.zeros([M,np.size(label_test,1)])
# pred_Euler = np.zeros([M,np.size(label_test,1)])
# pred_RK4 = np.zeros([M,np.size(label_test,1)])
# pred_PEC = np.zeros([M,np.size(label_test,1)])


# for k in range(0,M):
 
#     if (k==0):

#         out_direct = directstep(mynet_directstep,input_test_torch[0,:])
#         pred_direct [k,:] = out_direct.detach().cpu().numpy()

#         out_Euler = Eulerstep(mynet_Eulerstep,input_test_torch[0,:])
#         pred_Euler [k,:] = out_Euler.detach().cpu().numpy()

#         out_RK4 = RK4step(mynet_RK4step,input_test_torch[0,:])
#         pred_RK4 [k,:] = out_RK4.detach().cpu().numpy()

#         out_PEC = PECstep(mynet_RK4step,input_test_torch[0,:])
#         pred_PEC [k,:] = out_PEC.detach().cpu().numpy()

#     else:

#         out_direct = directstep(mynet_directstep,torch.from_numpy(pred_direct[k-1,:]).float().cuda())
#         pred_direct [k,:] = out_direct.detach().cpu().numpy()

#         out_Euler = Eulerstep(mynet_Eulerstep,torch.from_numpy(pred_Euler[k-1,:]).float().cuda())
#         pred_Euler [k,:] = out_Euler.detach().cpu().numpy()

#         out_RK4 = RK4step(mynet_RK4step,torch.from_numpy(pred_RK4[k-1,:]).float().cuda())
#         pred_RK4 [k,:] = out_RK4.detach().cpu().numpy()

#         out_PEC = PECstep(mynet_PECstep,torch.from_numpy(pred_PEC[k-1,:]).float().cuda())
#         pred_PEC [k,:] = out_PEC.detach().cpu().numpy()


# matfiledata_direct = {}
# matfiledata_direct[u'prediction'] = pred_direct
# matfiledata_direct[u'Truth'] = label_test 
# hdf5storage.write(matfiledata_direct, '.', path_outputs+'predicted_directstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

# matfiledata_Euler = {}
# matfiledata_Euler[u'prediction'] = pred_Euler
# matfiledata_Euler[u'Truth'] = label_test 
# hdf5storage.write(matfiledata_Euler, '.', path_outputs+'predicted_Eulerstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

# matfiledata_RK4 = {}
# matfiledata_RK4[u'prediction'] = pred_RK4
# matfiledata_RK4[u'Truth'] = label_test 
# hdf5storage.write(matfiledata_RK4, '.', path_outputs+'predicted_RK4step_1024_lead'+str(lead)+'.mat', matlab_compatible=True)

# matfiledata_PEC = {}
# matfiledata_PEC[u'prediction'] = pred_PEC
# matfiledata_PEC[u'Truth'] = label_test 
# hdf5storage.write(matfiledata_PEC, '.', path_outputs+'predicted_PECstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)



# print('Saved Predictions')

        
