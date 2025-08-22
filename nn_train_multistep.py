import numpy as np
import torch
import os
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_FNO import FNO1d
from nn_step_methods import *
from nn_spectral_loss import spectral_loss
from nn_jacobian_loss import *

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead, 'FNO')

net_name = 'FNO_RK4step_lead'+str(lead)+'_train_multistep_jacobian'
print(net_name)

chkpts_path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'
net_chkpt_path = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'

net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_RK4step_lead1_train_multistep_jacobian/chkpt_FNO_RK4step_lead1_train_multistep_jacobian_epoch_20.pt"
print(net_file_path)
starting_epoch = 21
print('Starting epoch '+str(starting_epoch))

if not os.path.exists(net_chkpt_path):
    os.makedirs(net_chkpt_path)
    print(f"Folder '{net_chkpt_path}' created.")
else:
    print(f"Folder '{net_chkpt_path}' already exists.")

with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000


epochs = 100
# batch_size = 50
batch_size = 100
batch_time = 5
batch_size_test = 100
batch_time_test = 30

print('Batch size ', batch_size)
print('Batch time: '+str(batch_time))
wavenum_init = 100
lamda_reg = 5
evalN = 10000
print('Batch time test: '+str(batch_time_test))

def Dataloader(data,batch_size,batch_time, key):
    time_chunks = []
    for i in range(data.shape[0] - batch_time*lead):
        time_chunks.append(data[i:i+batch_time*lead:lead])
    extra = len(time_chunks) % batch_size
    if extra==0:
        time_chunks = np.array(time_chunks)
    else:
        time_chunks = np.array(time_chunks[:-extra])
    rng = np.random.default_rng(key)
    split = rng.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    return split



device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 256  # input and output channels to the FNO layer

# learning_rate = 1e-5
learning_rate = 1.0941898913151241e-06
lr_decay = 0.4

mynet = FNO1d(modes, width, 1, 1).cuda()

step_net = RK4_step(mynet, device, time_step).to(device)

step_net.load_state_dict(torch.load(net_file_path))
print('state dict loaded')

count_parameters(mynet)

optimizer = optim.AdamW(step_net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

rng = np.random.default_rng()
key = rng.integers(100, size=1)
train_data = Dataloader(data[:,0:trainN+lead].T, batch_size = batch_size, batch_time = batch_time, key=key)
train_data = torch.from_numpy(train_data).float()

rng = np.random.default_rng()
key = rng.integers(100, size=1)
test_data = Dataloader(data[:,trainN+lead:].T, batch_size = batch_size_test, batch_time = batch_time_test, key=key)
test_data = torch.from_numpy(test_data).float()

class Loss_Multistep(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func

    def forward(self, batch): # batchsize x timesteps x space x c        
        x_i = self.model(batch[:,0])
        loss = self.loss_func(x_i, batch[:,1])
        loss.backward()
        for i in range(2, self.batch_time-1):
            # x_i = self.model(x_i.detach())
            x_i = self.model(batch[:,i-1])
            loss = self.loss_func(x_i, batch[:,i])
            loss.backward()
        return loss

class Loss_Multistep_Test(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func

    def forward(self, batch): # batchsize x timesteps x space x c        
        x_i = self.model(batch[:,0])
        loss = self.loss_func(x_i, batch[:,1])
        # loss.backward()
        for i in range(2, self.batch_time-1):
            x_i = self.model(x_i.detach())
            loss = self.loss_func(x_i, batch[:,i])
            # loss.backward()
        return loss

loss_fn = nn.MSELoss(reduction='mean')  #for basic loss func
loss_func = lambda e: torch.linalg.norm(e, dim=1).mean(0) 

# loss_net_jac = Trace_Jacobian_Loss_Multistep(step_net, batch_time, loss_func)
loss_net_jac = Jacobain_Loss_Multistep(step_net, batch_time, loss_func)
# loss_net_hes = Hessian_Train_Multistep(step_net, batch_time, loss_func)

# loss_net = Loss_Multistep(step_net, batch_time, loss_fn)
loss_net_test = Loss_Multistep_Test(step_net, batch_time_test, loss_fn)


torch.set_printoptions(precision=10)
best_loss = 1e5
for ep in range(starting_epoch, epochs+1):
    running_loss = 0.0
    for n in range(train_data.shape[0]):
        batch = train_data[n].unsqueeze(-1).to(device)
        optimizer.zero_grad()


        loss = loss_net_jac(batch)

        # loss = loss_net_hes(batch)

        # loss = loss_net(batch)


        # loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()
        # print(loss)

    scheduler.step()

    net_loss = (running_loss/(train_data.shape[0]))
    key = np.random.randint(len(test_data))
    with torch.no_grad():
        test_loss = loss_net_test(test_data[key].unsqueeze(-1).to(device))
    print(f'Epoch : {ep}, Train Loss : {net_loss/(batch_time-1)}, Test Loss : {test_loss/(batch_time_test-1)}')
    print('Learning rate', scheduler._last_lr)
    # print('Time_step_val:'+str(step_net.time_step))
    
    if best_loss > test_loss:
        print('Saved!!!')
        torch.save(step_net.state_dict(), chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'.pt')
        print('Checkpoint updated')
        print(chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'.pt')
        best_loss = test_loss

    if ep % 10 == 0:
        print(chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')
        torch.save(step_net.state_dict(), chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')

torch.save(step_net.state_dict(), net_chkpt_path+'chkpt_'+net_name+'_final.pt')
torch.set_printoptions(precision=4)
print("Model Saved")
