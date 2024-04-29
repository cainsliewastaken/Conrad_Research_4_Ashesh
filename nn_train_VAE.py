import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_VAE import VAE
from nn_spectral_loss import spectral_loss

def Eulerstep(net, input_batch, time_step):
 output_1, mu, sigma = net(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1), mu, sigma
  
def Directstep(net, input_batch, time_step):
  output_1, mu, sigma = net(input_batch.cuda())
  return output_1, mu, sigma

def RK4step(net, input_batch, time_step):
 output_1, mu1, sigma1 = net(input_batch.cuda())
 output_2, mu2, sigma2 = net(input_batch.cuda()+0.5*output_1)
 output_3, mu3, sigma3 = net(input_batch.cuda()+0.5*output_2)
 output_4, mu4, sigma4 = net(input_batch.cuda()+output_3)
 output_full = input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6
 mu_full = torch.mean(torch.stack([mu1, mu2, mu3, mu4], dim=0), dim=0)
 sigma_full = torch.mean(torch.stack([sigma1, sigma2, sigma3, sigma4], dim=0), dim=0)
 return output_full, mu_full, sigma_full

def PECstep(net, input_batch, time_step):
 output_1, mu, sigma = net(input_batch.cuda()) + input_batch.cuda()
 return input_batch.cuda() + time_step*0.5*(net(input_batch.cuda())+net(output_1)), mu, sigma

def PEC4step(net, input_batch, time_step):
 output_0, mu1, sigma1 = net(input_batch.cuda())
 output_1 = time_step*(output_0) + input_batch.cuda()
 output_2, mu2, sigma2 = net(output_1)
 output_2 = input_batch.cuda() + time_step*0.5*(output_0+output_2)
 output_3, mu3, sigma3 = net(output_2)
 output_3 = input_batch.cuda() + time_step*0.5*(output_0+output_3)
 output_4, mu4, sigma4 = net(output_3)
 output_4 = time_step*0.5*(output_0+output_4)
 mu_full = torch.mean(torch.stack([mu1, mu2, mu3, mu4], dim=0), dim=0)
 sigma_full = torch.mean(torch.stack([sigma1, sigma2, sigma3, sigma4], dim=0), dim=0)
 return input_batch.cuda() + output_4, mu_full, sigma_full

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead, 'VAE')

chkpts_path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'

step_func = Directstep
print(step_func)

net_name = 'VAE_Directstep_lead'+str(lead)+''

net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/NN_FNO_PEC4step_lead1_tendency/chkpt_NN_FNO_PEC4step_lead1_tendency_epoch51.pt"

# to change from normal loss to spectral loss scroll down 2 right above train for loop

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])


trainN = 150000
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).cuda()
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).cuda()
du_label_torch = (input_train_torch - label_train_torch).cuda()

device = 'cuda'  #change to cpu if no cuda available

#model parameters
input_size = 1024
output_size = 1024
hidden_layer_size = 2000
distribution_size = 256
learning_rate = 0.001
lr_decay = 0.4

mynet = VAE(input_size, output_size, hidden_layer_size, distribution_size).cuda()
# mynet.load_state_dict(torch.load(net_file_path))

count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)

epochs = 60
batch_size = 25
wavenum_init = 100
lambda_reg = 5
beta = .1
print('Beta is ', beta)

loss_fn = nn.MSELoss()  #for basic loss func
loss_fc = spectral_loss #for spectral loss in tendency, also change loss code inside for loop below
torch.set_printoptions(precision=10)

for ep in range(0, epochs+1):
    for step in range(0,trainN,batch_size):
        indices = np.random.permutation(np.arange(start=step, step=1,stop=step+batch_size))
        input_batch, label_batch, du_label_batch = input_train_torch[indices].cuda(), label_train_torch[indices].cuda(), du_label_torch[indices].cuda()
        #pick a random boundary batch
        
        
        optimizer.zero_grad()
        outputs, mu, sigma = step_func(mynet, input_batch.float(), time_step)
        kl_divergence = 0.5 * torch.sum(-1 - sigma + mu.pow(2) + sigma.exp())
        loss = loss_fn(outputs, label_batch.float()) + beta*kl_divergence # use this loss function for mse loss

        # outputs_2, mu2, sigma2 = step_func(mynet, outputs, time_step) #use these lines below for spectral loss
        # kl_divergence_2 = 0.5 * torch.sum(-1 - sigma2 + mu2.pow(2) + sigma2.exp())
        # loss = loss_fc(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)
        # loss = loss + beta*kl_divergence + beta*kl_divergence_2

        loss.backward()
        

        optimizer.step()


    if ep % 1 == 0:
        print('Epoch', ep)
        print ('Loss', loss)
        torch.save(mynet.state_dict(), chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')


torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")