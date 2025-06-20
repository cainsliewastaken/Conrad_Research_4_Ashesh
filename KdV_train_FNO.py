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
from nn_step_methods import *
from nn_spectral_loss import spectral_loss
from nn_jacobian_loss import *

def load_output_data(num_chunks, file_path, M,):

    output_data = np.zeros([int(M),input_size])
    # output_data_RMSE = np.zeros([int(M)+1])
    # output_data_FFT_X = np.zeros([int(M)+1,512])

    start_ind = 0
    for k in range(num_chunks):
        out = np.load(file_path+str(k)+'.npz', allow_pickle=True)#[10000:,:,:,:]
        t_range = out['arr_0'].shape[1]
        # print(out['arr_0'].shape)
        output_data[start_ind:start_ind+t_range] = out['arr_0'].T
        # output_data_RMSE[start_ind:start_ind+t_range] = out.item()['RMSE']
        # output_data_FFT_X[start_ind:start_ind+t_range,:,:512] = out.item()['prediction'][:,:,:512]
        start_ind += t_range
        # print(start_ind)

    return torch.from_numpy(output_data).float()#, torch.from_numpy(output_data_RMSE).float(), torch.from_numpy(output_data_FFT_X).float()


# L = 2*np.pi
L = 20
m = 512
x = np.arange(-m/2,m/2)*(L/m)

time_step = 1.73/((m/2)**3)
lead = 0
print(lead, 'FNO')

chkpts_path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'

net_name = 'KdV_FNO'
print(net_name)

# net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Directstep_lead50/chkpt_FNO_Directstep_lead50_epoch53.pt"
# print(net_file_path)
starting_epoch = 0
print('Starting epoch '+str(starting_epoch))

# to change from normal loss to spectral loss scroll down 2 right above train for loop

M = 1000000
trainN = int(M*.7)
input_size = 256
output_size = 256
hidden_layer_size = 2000

num_chunks = 10
file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KdV_data_spectral_delta_3_nu_0/KdV_data_spectral_delta_3_nu_0_chunk_"
output_data = load_output_data(num_chunks, file_path, M)
print(output_data.shape)

num_chunks = 10
file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KdV_data_spectral_delta_3_nu_0.1/KdV_data_spectral_delta_3_nu_0.1_chunk_"
output_data_nu = load_output_data(num_chunks, file_path, M)


input_train_torch = ((output_data[:,0:trainN])).cuda()
diff_train = torch.diff(input_train_torch, dim=0)

label_train_torch = ((output_data_nu[:,lead:lead+trainN])).cuda()
diff_label = torch.diff(label_train_torch, dim=0)


time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 128 # number of Fourier modes to multiply
width = 128  # input and output channels to the FNO layer

learning_rate = 1e-4
lr_decay = 0.4

mynet = FNO1d(modes, width, time_future, time_history).cuda()

# mynet.load_state_dict(torch.load(net_file_path))
# print('state dict loaded')

step_net = Direct_step(mynet, device, time_step)

count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)

epochs = 60
batch_size = 50
# batch_size = 100
print('Batch size ', batch_size)
wavenum_init = 100
lamda_reg = 5

loss_fn = nn.MSELoss()  #for basic loss func
# loss_fc = spectral_loss #for spectral loss in tendency, also change loss code inside for loop below
torch.set_printoptions(precision=10)

for ep in range(starting_epoch, epochs+1):
    running_loss = 0
    indices = np.random.permutation(torch.arange(trainN))
    for step in range(0,trainN,batch_size):
        batch_indices = indices[step:step + batch_size]
        # indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch = input_train_torch[batch_indices].cuda(), label_train_torch[batch_indices].cuda()
        input_batch = torch.reshape(input_batch,(batch_size,input_size,1)).float()
        label_batch = torch.reshape(label_batch,(batch_size,input_size,1)).float()
        # du_label_batch = du_label_torch[indices].cuda()
        # du_label_batch = torch.reshape(du_label_batch,(batch_size,input_size,1)).float()
        #pick a random boundary batch
        
        optimizer.zero_grad()
        outputs = step_net(input_batch)
        
        loss = loss_fn(outputs, label_batch)  # use this loss function for mse loss

        # outputs_2 = step_func(mynet, outputs, time_step) #use this line and line below for spectral loss
        # loss = loss_fc(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        # loss = jacobian_loss(step_net, input_batch, label_batch)
        # loss = spectral_jacobian_loss(step_net, input_batch, label_batch, 1, 1)

        loss.backward()

        optimizer.step()
        running_loss += loss.clone().detach()

    if ep % 1 == 0:
        print('Epoch', ep)
        print ('Train Loss', float(running_loss/int(trainN/batch_size)))
        with torch.no_grad():
            key = np.random.randint(0, trainN, 100)
            temp_loss = F.mse_loss(step_net(input_train_torch[key].reshape(100,input_size,1).cuda().float()), label_train_torch[key].reshape(100,input_size,1).cuda().float())
            print('One step loss:', float(temp_loss))

        torch.save(mynet.state_dict(), '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')


torch.save(mynet.state_dict(), net_name+'.pt')
torch.set_printoptions(precision=4)
print("Model Saved")

