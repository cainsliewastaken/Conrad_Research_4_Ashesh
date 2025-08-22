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

time_step = 1e-1
lead = int((1/1e-3)*time_step)
print(lead, 'FNO')

net_name = 'FNO_Third_order_3rd_udot_w_input_lead'+str(lead)+'_train_multistep_w_pullback_w_double'
print(net_name)

chkpts_path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'
net_chkpt_path = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'

# net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_lead100_train_multistep_hessian/chkpt_FNO_Eulerstep_lead100_train_multistep_hessian.pt"
# print(net_file_path)
starting_epoch = 0
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
testN = trainN + 50000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000


epochs = 100
# batch_size = 50
batch_size = 100
batch_time = 5
print('Batch size ', batch_size)
print('Batch time: '+str(batch_time))
wavenum_init = 100
lamda_reg = 5
evalN = 10000
batch_time_test = 30
batch_size_test = 400
print('Batch time test: '+str(batch_time_test))
print('Batch size test: '+str(batch_size_test))


def Dataloader_many_dts(data,batch_size, batch_time, key, num_prev_points):
    time_chunks = []
    for i in range(data.shape[0] - (num_prev_points +batch_time)*lead):
        data_chunk = data[i:i+(num_prev_points + batch_time)*lead:lead]
        if len(data_chunk) == num_prev_points + batch_time:
            time_chunks.append(data_chunk)
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

learning_rate = 1e-4
print(learning_rate)
# learning_rate = 1.9371148445850093e-06
lr_decay = 0.4

mynet = FNO1d(modes, width, 1, 3).cuda()

# mynet.load_state_dict(torch.load(net_file_path))
# print('state dict loaded')

# diff_var =  (data[:,2*lead:trainN:lead].T - 2*data[:,lead:trainN-lead:lead].T + data[:,0*lead:trainN-2*lead:lead].T).var()
# diff_var =  (data[:,3*lead:trainN:lead].T - 5/2*data[:,2*lead:trainN-lead:lead].T + 2*data[:,1*lead:trainN-2*lead:lead].T - 1/2*data[:,0*lead:trainN-3*lead:lead].T).var()
diff_var = (data[:,4*lead:trainN:lead].T - 17/6*data[:,3*lead:trainN-1*lead:lead].T + 3*data[:,2*lead:trainN-2*lead:lead].T -3/2*data[:, 1*lead:trainN-3*lead:lead].T + 1/3*data[:,0*lead:trainN-4*lead:lead].T -2*data[:,3*lead:trainN-1*lead:lead].T + 5*data[:,2*lead:trainN-2*lead:lead].T -4*data[:, 1*lead:trainN-3*lead:lead].T + 1*data[:,0*lead:trainN-4*lead:lead].T).var()

print(diff_var)

step_net = Third_order_multistep_3rd_order_udot(mynet, device, diff_var).to(device)

count_parameters(step_net)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
# optimizer_time_step = optim.AdamW(step_net.time_step, lr=1e-8)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

rng = np.random.default_rng()
key = rng.integers(100, size=1)
train_data = Dataloader_many_dts(data[:,0:trainN+lead].T, batch_size = batch_size, batch_time = batch_time, key=key, num_prev_points=4)
train_data = torch.from_numpy(train_data)
print('Loaded training dataset')

rng = np.random.default_rng()
key = rng.integers(100, size=1)
test_data = Dataloader_many_dts(data[:,trainN+lead:testN].T, batch_size = batch_size_test, batch_time = batch_time_test, key=key, num_prev_points=4)
test_data = torch.from_numpy(test_data)
print('Loaded eval dataset')

class Loss_Multistep(nn.Module):
    def __init__(self, model, batch_time, loss_func, num_prev_points):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func
        self.num_prev_points = num_prev_points

    def forward(self, batch): # batchsize x timesteps x space x c        
        x_i = self.model(batch[:, 0:self.num_prev_points])
        loss = self.loss_func(x_i, batch[:, self.num_prev_points])
        loss.backward()

        x_prev_vals = batch[:, 1:self.num_prev_points]
        for i in range(2, self.batch_time-1):
            x_vals = torch.cat([x_prev_vals, x_i.unsqueeze(1)], dim=1)
            x_prev_vals = x_vals[:,1:]
            x_i = self.model(x_vals.detach())
            # x_i = self.model(batch[:,i-1])
            loss = self.loss_func(x_i, batch[:, i + self.num_prev_points - 1])
            loss.backward()
        return loss

class Loss_Multistep_Test(nn.Module):
    def __init__(self, model, batch_time, loss_func, num_prev_points):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func
        self.num_prev_points = num_prev_points

    def forward(self, batch): # batchsize x timesteps x space x c        
        x_i = self.model(batch[:, 0:self.num_prev_points])
        loss = self.loss_func(x_i, batch[:, self.num_prev_points])

        x_prev_vals = batch[:, 1:self.num_prev_points]
        for i in range(2, self.batch_time-1):
            x_vals = torch.cat([x_prev_vals, x_i.unsqueeze(1)], dim=1)
            x_prev_vals = x_vals[:,1:]
            x_i = self.model(x_vals)
            # x_i = self.model(batch[:,i-1])
            loss = self.loss_func(x_i, batch[:, i + self.num_prev_points - 1])

        return loss
    

loss_fn = nn.MSELoss(reduction='mean')  #for basic loss func
loss_func = lambda e: torch.linalg.norm(e, dim=1).mean(0) 

# loss_net_jac = Trace_Jacobian_Loss_Multistep(step_net, batch_time, loss_func)
# loss_net_jac = Jacobain_Train_Multistep(step_net, batch_time, loss_func)
# loss_net_hes = Hessian_Train_Multistep(step_net, batch_time, loss_func)

loss_net = Loss_Multistep(step_net, batch_time, loss_fn, num_prev_points=4)
loss_net_test = Loss_Multistep_Test(step_net, batch_time_test, loss_fn, num_prev_points=4)
print('Starting train')

torch.set_printoptions(precision=10)
best_loss = 1e5
for ep in range(starting_epoch, epochs+1):
    running_loss = 0.0
    for n in range(train_data.shape[0]):
        batch = train_data[n].unsqueeze(-1).to(device)
        optimizer.zero_grad()
        # optimizer_time_step.zero_grad()

        # loss = loss_net_jac(batch)

        # loss = loss_net_hes(batch)

        loss = loss_net(batch)

        # loss.backward()
        optimizer.step()
        # optimizer_time_step.step()
        running_loss += loss.detach().item()
        # print(loss)

    scheduler.step()

    net_loss = (running_loss/(train_data.shape[0]))
    key = np.random.randint(len(test_data))
    with torch.no_grad():
        test_loss = loss_net_test(test_data[key].unsqueeze(-1).to(device))
    print(f'Epoch : {ep}, Train Loss : {net_loss/(batch_time-1)}, Test Loss : {test_loss/(batch_time_test-1)}')
    print('Learning rate', scheduler._last_lr)
    print('Time_step_val:'+str(step_net.time_step.item()))
    
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
