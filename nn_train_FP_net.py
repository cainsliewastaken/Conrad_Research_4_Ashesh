import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import pickle
import torch
from Obs_func_generator import EnsKFstep_module
from nn_step_methods import *
import torch.optim as optim
from nn_MLP import *
from nn_FNO import *

noise_var = 3
ensemble_size = 400
ydim = 512

chkpts_path = '/glade/derecho/scratch/cainslie/conrad_net_stability/FP_chkpts/'
model_name = 'KS_FP_net_MLP_'+str(noise_var)+'_noise_'+str(ensemble_size)+'_ens_'+str(ydim)+'_ydim'
print(model_name)


starting_epoch = 31

print(starting_epoch)

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

lead=1
time_step = 1e-3
trainN= 150000
# trainN = 25000
testN = 50000

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver

#model parameters
modes = 256 # number of Fourier modes to multiply
# ORIGINAL: width = 64 # input and output chasnnels to the FNO layer
width = 256

# mynet = FNO1d(modes, width, time_future, time_history).cuda()
# mynet.load_state_dict(torch.load('/glade/derecho/scratch/cainslie/conrad_net_stability/Conrad_Research_4_Ashesh/FNO_PEC4step_lead1_tendency.pt'))
# mynet.cuda()

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda() #NEED TO KEEP AS numpy!!
# label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:trainN+testN])).float().cuda()
# label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()

#FPmodel parameters
time_chunk_size = 0
input_size = 1024
learning_rate = 1e-5
learning_rate = 3.4867844010000007e-06
batch_size = 20
test_batch_size = 32
step_method = PEC4step

print('time chunk size:',time_chunk_size,'\n ensemble_size:',ensemble_size,'\n batch size:',batch_size)
print('noise var: ', noise_var)
print('Learning rate:', learning_rate)
print('TrainN:',trainN)

print('Mem before model dec',torch.cuda.memory_allocated())

#EnsKFstep_module parameters
hidden_size = input_size*4
observation_func = MLP_net_variable(input_size, ydim, hidden_size, 8, activation=F.tanh, use_act=False).cuda() #Made it only 3 hidden layers (HIDDEN LAYERS WERE 512, now 1024)

EnsKFstep = EnsKFstep_module(observation_func,ydim).cuda()
FP_model = FPmodel_parent(EnsKFstep, time_chunk_size, step_method, batch_size, ensemble_size, time_step, 1024, noise_var).cuda()
FP_model.load_state_dict(torch.load('/glade/derecho/scratch/cainslie/conrad_net_stability/FP_chkpts/KS_FP_net_MLP_3_noise_400_ens_512_ydim/chkpt_KS_FP_net_MLP_3_noise_400_ens_512_ydim_epoch_30.pt'))

optimizer = optim.AdamW(FP_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

print('Mem after model dec ',torch.cuda.memory_allocated())
print('Mem usage frac before backprop ', torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory)

best_loss = 1e5
epochs = 50
print('Starting training')

num_batches_per_epcoh = int(np.ceil((trainN - time_chunk_size)/batch_size))
print(num_batches_per_epcoh,'Batches per epoch')

for epoch in range(starting_epoch, epochs):
    # Shuffle the data and create batches for this epoch
    indices = np.random.permutation(torch.arange(trainN - time_chunk_size))
    loss_avg = 0

    for batch_start in range(0, trainN - time_chunk_size, batch_size): #sampling w/ removal
        # Select the batch indices for this step
        batch_indices = indices[batch_start:batch_start + batch_size]
        input_batch = input_train_torch[batch_indices, :]
        label_batch = input_train_torch[batch_indices + time_chunk_size, :]

        optimizer.zero_grad()

        train_output = FP_model(input_batch)
        loss = FP_model.loss_function(train_output, label_batch).mean()
        loss.backward()
        # print(loss)

        optimizer.step()
        
        loss_avg += loss.clone().detach()
        
        # if epoch == 0:
            # print(float(loss))
            # print(float(FP_model.EnsKF_step_module.y_true_mean.norm()), '\n',float(FP_model.EnsKF_step_module.y_true_std.norm()))

    loss_avg = loss_avg / num_batches_per_epcoh
    scheduler.step()
    
    print(float(FP_model.EnsKF_step_module.y_true_mean.norm()), float(FP_model.EnsKF_step_module.y_true_std.norm()))

    with torch.no_grad(): #VALIDATION, going over the entire set makes no sense. Just sample a few points and mean
        test_indices = torch.randint(0, input_test_torch.shape[0], size = (test_batch_size,1)).squeeze(1)
        input_batch_test = input_test_torch[test_indices, :]
        label_batch_test = input_test_torch[test_indices + time_chunk_size, :]

        test_output = FP_model(input_batch_test)

        test_loss_avg = FP_model.loss_function(test_output, label_batch_test).mean()

        print(f'Epoch: {epoch}, Train loss: {loss_avg}, Validation loss: {test_loss_avg}, Learning rate: {scheduler.get_last_lr()}')
        
        if test_loss_avg < best_loss:
            print('best loss saved!')
            best_loss = test_loss_avg
            torch.save(FP_model.state_dict(),chkpts_path+model_name+'/chkpt_'+model_name+'.pt')

    if epoch % 5 == 0:
        torch.save(FP_model.state_dict(), chkpts_path+model_name+'/chkpt_'+model_name+'_epoch_'+str(epoch)+'.pt')


torch.save(FP_model.state_dict(),chkpts_path+model_name+'/best_'+ model_name +'.pt')
