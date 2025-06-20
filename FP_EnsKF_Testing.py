import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import pickle
import torch
from Obs_func_generator import EnsKFstep_module
#from eval_plot_FNO_tendency import FNO1d
from nn_step_methods import RK4step
import torch.optim as optim
import matplotlib.pyplot as plt
from obs_func_Hx import MLP_Net
from FP_parent_class import SpectralConv1d
from FP_parent_class import FNO1d
from FP_parent_class import FPmodel_parent
print(torch.cuda.is_available())

with open('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl', 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

lead=1
time_step = 1e-3
#trainN= 150000
trainN = 10000

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver

#model parameters
modes = 256 # number of Fourier modes to multiply
# ORIGINAL: width = 64 # input and output chasnnels to the FNO layer
width = 256

mynet = FNO1d(modes, width, time_future, time_history).cuda()
mynet.load_state_dict(torch.load('/glade/derecho/scratch/cainslie/conrad_net_stability/Conrad_Research_4_Ashesh/FNO_PEC4step_lead1_tendency.pt'))
mynet.cuda()

input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN])).float().cuda() #NEED TO KEEP AS numpy!!
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN])).float().cuda()

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float().cuda()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead:])).float().cuda()

#FPmodel parameters
time_chunk_size = 10
input_size = 1024
learning_rate = 0.0005
ensemble_size = 50
batch_size = 1
step_method = RK4step

#EnsKFstep_module parameters
ydim = int(input_size/8)
hidden_size = input_size
observation_func = MLP_Net(input_size,hidden_size,ydim).cuda()

EnsKFstep= EnsKFstep_module(observation_func,ydim).cuda()
FP_model = FPmodel_parent(mynet, EnsKFstep, time_chunk_size, step_method, batch_size, ensemble_size, time_step, input_size=1024, noise_std=0.1).cuda()
# FP_model.load_state_dict(torch.load('/media/volume/FP_EnsKF_training/NN_files/FP_EnsKF_model_state_dict_DIFFR3_BEST_LOSS_epoch_.pt'))
FP_model.cuda()

#optimizer = optim.AdamW(FP_model.parameters(), lr=learning_rate)

#Prediction Parameters
pred_len = 500 #length of prediction on Test set
test_output = torch.zeros(ensemble_size, input_size, pred_len)
noise_std = 3
#print('Starting epoch'+str(epoch))

print('Starting Prediction')
for t in range(0, pred_len, time_chunk_size):
    print(t)
    if (t == 0):
        print('input test torch[0,:] shape 1024, max/min:',input_test_torch[0,:].max(),input_test_torch[0,:].min())
        first_step = input_test_torch[0,:].unsqueeze(0).cuda() + (torch.randn(ensemble_size, input_size)*noise_std).cuda()

        print('first step max all',first_step.max())
        print('first step 5th ens max/min',first_step[5,:].max(),first_step[5,:].min())
        print('first step after noise',np.shape(first_step))

        # Check if first_step and test_output are on GPU
        assert first_step.is_cuda, "first_step is not on CUDA"
        #assert test_output.is_cuda, "test_output is not on CUDA"
        #first_step = first_step.unsqueeze(0)
        #print('first step after squeeze', np.shape(first_step))
        pred = FP_model.inference(first_step) #ensem x inputsize x predlen
        print('shape first pred', np.shape(pred))
        test_output[:,:,0:time_chunk_size] = pred.squeeze(0).detach().cpu()

    else:
        if t % 50 == 0:
            print(t)
        print('test output[1,:,t-1] max',test_output[1,:,t-1].max())
        pred = FP_model.inference(test_output[:,:,t-1])
        print('shape test_output in else', np.shape(test_output))
        print('pred max and min [0,5,:,t]',pred[0,5,:,5].max(),pred[0,:,1].min())
        test_output[:,:,t:t+time_chunk_size] = pred.squeeze(0).detach().cpu()

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#test_output= torch.reshape(test_output,(pred_len,ensemble_size,input_size))
test_output = test_output.permute(2,0,1)
print('shape test output after reshape', np.shape(test_output))
debug_frame = test_output[0,10,:] #0th timestep, 10th ensembles, all spatial
single_prediction = torch.mean(test_output,1)

#print('single pred max',single_prediction.max())
print('shape single prediction',np.shape(single_prediction))

single_prediction = single_prediction.detach().cpu().numpy()
truth_ = input_test_torch[:pred_len,:].cpu().numpy()

vmin = min(truth_.min(), single_prediction.min())
vmax = max(truth_.max(), single_prediction.max())
print('min and max single pred',single_prediction.min(), single_prediction.max())

# Create a figure with two subplots (1 row, 2 columns)
fig, ax = plt.subplots(2, 4, figsize=(12, 6))  # 1 row, 2 columns

# Plot the "truth" image and get the imshow object
im_truth = ax[0,0].imshow(truth_, cmap='viridis', aspect = 'auto', vmin=vmin, vmax=vmax)
ax[0,0].set_title('Truth')  # Label the first subplot
fig.colorbar(im_truth, ax=ax[0,0])  # Add color bar to the truth plot

# Plot the "prediction" image and get the imshow object
im_pred = ax[0,1].imshow(single_prediction, cmap='viridis', aspect = 'auto', vmin=vmin, vmax=vmax)
ax[0,1].set_title('Prediction')  # Label the second subplot
fig.colorbar(im_pred, ax=ax[0,1])  # Add color bar to the prediction plot

ax[0,2].plot(range(np.shape(single_prediction)[1]),single_prediction[400,:])
ax[0,2].set_title('ensem avg prediction at timestep 400')
ax[0,3].plot(range(np.shape(truth_)[1]), truth_[400,:])
ax[0,3].set_title('truth at timestep 400')

ax[1,0].plot(range(np.shape(single_prediction)[1]),single_prediction[40,:])
ax[1,0].set_title('ensem avg prediction at timestep 40')
ax[1,1].plot(range(np.shape(truth_)[1]), truth_[40,:])
ax[1,1].set_title('truth at timestep 40')

# Save the plots to a file
plt.tight_layout()  # Adjust spacing between subplots
plt.savefig('Truth_vs_prediction.png')  # Save the figure to a PNG file

# matfiledata_FP_EnsKF_RK4 = {}
# matfiledata_FP_EnsKF_RK4[u'prediction'] = test_output
# matfiledata_FP_EnsKF_RK4[u'Truth'] = label_test_torch
# matfiledata_FP_EnsKF_RK4[u'RMSE'] = RMSE(test_output, label_test_torch[:,:np.shape(pred_RK4)[1],:])
# scipy.io.savemat(path_outputs+'batch_'+str(job_id)+'_predicted_RK4step_1024_FNO_tendency_lead'+str(lead)+'_ensemble2_.mat', matfiledata_RK4)
# #hdf5storage.write(matfiledata_Euler, '.', path_outputs+'predicted_Eulerstep_1024_lead'+str(lead)+'.mat', matlab_compatible=True)
# print('Saved RK4 prediction')





#every ten epochs get loss for test data to see if we are overfitting
#implement scheduler, decreases learning rate every epoch
#every epoch  take learning 
