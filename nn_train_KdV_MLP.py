import numpy as np
import torch
print(torch.__version__)
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
# from count_trainable_params import count_parameters
import pickle
from nn_MLP import MLP_Net, MLP_net_variable
from nn_step_methods import Directstep, Eulerstep, RK4step, PECstep, PEC4step
from nn_spectral_loss import spectral_loss
# from nn_Cascade_MLP import Cascade_MLP_Net

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead, 'FNO')

chkpts_path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'

step_func = Directstep
print(step_func)

net_name = 'MLP_KdV_x_grad_net'
print(net_name)

# net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_PEC4step_lead1_tendency/chkpt_FNO_PEC4step_lead1_tendency_epoch27.pt"
starting_epoch = 0

# to change from normal loss to spectral loss scroll down 2 right above train for loop
def load_output_data(num_chunks, file_path, M,):

    output_data = np.zeros([int(M),16])
    # output_data_RMSE = np.zeros([int(M)+1])
    # output_data_FFT_X = np.zeros([int(M)+1,512])

    start_ind = 0
    for k in range(0, num_chunks):
        out = np.load(file_path+str(k)+'.npz', allow_pickle=True)#[10000:,:,:,:]
        t_range = out['arr_0'].shape[1]
        # print(out['arr_0'].shape)
        output_data[start_ind:start_ind+t_range] = out['arr_0'].T
        # output_data_RMSE[start_ind:start_ind+t_range] = out.item()['RMSE']
        # output_data_FFT_X[start_ind:start_ind+t_range,:,:512] = out.item()['prediction'][:,:,:512]
        start_ind += t_range
        # print(start_ind)
        if k % 10 == 0: print(k)

    return output_data

num_chunks = 100
M = 10000000
file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KdV_data_spectral_alpha_5_delta_1_nu_0.001_long/KdV_data_spectral_alpha_5_delta_1_nu_0.001_long_chunk_"
data = load_output_data(num_chunks, file_path, M)
data=np.asarray(data[:,:])

L = 2*np.pi
# L = 20
m = 16
x = np.arange(-m/2,m/2)*(L/m)
dx = np.abs(x[1] - x[0])
dt = 1e-5
kx = np.fft.fftfreq(m)*m*2*np.pi/L

#----- Training data Parameters -----
alpha = 5
delta3 = 1
nu = 0.001



batch_size = 40
batch_time = 40
trange = 1000000

# data = np.arange(1010*16).reshape(1010,16)

u = data[0:trange+batch_time-1]
ux = np.gradient(u, dx, axis=1) 
uxx = np.gradient(ux, dx, axis=1) 
uxxx = np.gradient(uxx, dx, axis=1) 

label_train_torch = torch.from_numpy(np.stack([u, ux, uxx, uxxx], axis=-1)).cuda()


def Dataloader(u,batch_size,batch_time, key):
    time_chunks = []
    for i in range(u.shape[0] - batch_time):
        time_chunks.append(u[i:i+batch_time])
    extra = len(time_chunks) % batch_size
    if extra==0:
        time_chunks = np.array(time_chunks)
    else:
        time_chunks = np.array(time_chunks[:-extra])
    print(time_chunks.shape)
    split = np.array(np.split(time_chunks, time_chunks.shape[0]//batch_size))
    rng = np.random.default_rng(key)
    indexs = rng.permutation(np.arange(split.shape[0]))
    split = split[indexs]

    return split, indexs

rng = np.random.default_rng()
key = rng.integers(100, size=1)
train_data, indexs = Dataloader(u, batch_size = batch_size, batch_time = batch_time, key=key)

label_train_torch = np.array(np.split(label_train_torch[batch_time-1:].cpu(), label_train_torch[batch_time-1:].shape[0]//batch_size))

label_train_torch = torch.Tensor(label_train_torch[indexs]).float()



input_train_torch = torch.from_numpy((train_data)).cuda().float()

input_train_torch = input_train_torch[:,:,:,8]
label_train_torch = label_train_torch.cuda()[:,:,8,1:]


input_train_torch_mean = input_train_torch.mean()
input_train_torch_std = input_train_torch.std()
input_train_torch = (input_train_torch - input_train_torch_mean)/input_train_torch_std



label_train_torch_mean = label_train_torch.mean()
label_train_torch_std = label_train_torch.std()
label_train_torch = (label_train_torch - label_train_torch_mean)/label_train_torch_std


print(input_train_torch.shape, label_train_torch.shape)


hidden_layer_size = 2056

# mynet = MLP_Net(batch_time, hidden_layer_size, 3).cuda()
mynet = MLP_net_variable(batch_time, 3, hidden_layer_size, 5, use_act=False, use_dropout=False).cuda()

learning_rate = 0.0001
lr_decay = 0.4
optimizer = optim.Adam(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)



loss_fn = nn.MSELoss()
epochs = 120
batch_size = 100
wavenum_init = 100
lamda_reg = 5

torch.set_printoptions(precision=10)

if not os.path.exists(chkpts_path_outputs+str(net_name)+'/'):
    os.makedirs(chkpts_path_outputs+str(net_name)+'/')
    print(f"Folder '{chkpts_path_outputs+str(net_name)+'/'}' created.")
else:
    print(f"Folder '{chkpts_path_outputs+str(net_name)+'/'}' already exists.")

for ep in range(0, epochs+1):
    for step in range(train_data.shape[0]):
        input_batch, label_batch = input_train_torch[step], label_train_torch[step]
        #pick a random boundary batch
        optimizer.zero_grad()
        outputs = step_func(mynet, input_batch.float(), time_step)
        
        loss = loss_fn(outputs, label_batch.float()) #use this for basic mse loss 

        loss.backward()
        optimizer.step()
    scheduler.step()

    if ep % 5 == 0:
        print('Epoch', ep)
        print ('Loss', loss)
        torch.save(mynet.state_dict(), chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')

torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")