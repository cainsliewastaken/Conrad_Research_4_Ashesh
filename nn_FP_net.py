import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F


class FPmodel_parent(nn.Module):
    def __init__(self, EnsKF_step_module, time_chunk_size, step_method, batch_size, ensemble_size, timestep, input_size, noise_var):
        super(FPmodel_parent, self).__init__()
        # self.myNet = myNet
        self.EnsKF_step_module = EnsKF_step_module 
        self.time_chunk_size = time_chunk_size #time steps before EnsKF step
        self.step_method = step_method
        self.input_size = input_size 
        self.noise_var = noise_var 
        self.batch_size = batch_size 
        self.ensemble_size =ensemble_size #not currently being used because batchsize = real_batchsize * ensemblesize
        self.timestep = timestep

    def forward(self, x): #the x passed here should only be the train_input, not the train_label
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.ensemble_size, self.input_size)
        ens_tens = x.unsqueeze(1) + (torch.randn(batch_size, self.ensemble_size, self.input_size)*self.noise_var).cuda() #x is batch x 1024, I need batch x 1 x 1024
        output = ens_tens

        # output = torch.reshape(output,(batch_size * self.ensemble_size, self.input_size,1)) #reshape to batches*ensembles x inputsize before progressing a step
        # # with torch.no_grad(): #no gradients for going forward 50=(chunk_size) time steps using myNet_stepmethod
        # #     for step in range(self.time_chunk_size): #run for however many time steps are specified by time_chunk_size
        # #         output = self.step_method(self.myNet,output,self.timestep)
        # output = torch.reshape(output,(batch_size, self.ensemble_size, self.input_size)) #return back to batchsize x ensembles x inputsize
        
        output = self.EnsKF_step_module(output) #takes last prediction and corrects it ALSO: has gradients stored
        return output #batchsize x ensembles x inputsize

    def inference(self, x): #the x passed here should only be the train_input, not the train_label
        #print('shape x',np.shape(x))
        output = torch.zeros(self.ensemble_size, self.input_size, self.time_chunk_size).cuda()
        #print('output shape in inference', np.shape(output))
        ens_tens = x # batchsize=1 x ensem x inputsize x timesteps
        ens_tens = torch.reshape(ens_tens,(self.batch_size * self.ensemble_size, self.input_size,1)) #reshape to 1*ensembles x inputsize x 1 before progressing a step
        #print('shape of ens_tens after reshape',np.shape(ens_tens))
        with torch.no_grad(): #no gradients for going forward (chunk_size)=10 time steps using myNet_stepmethod
            pred = self.step_method(self.myNet, ens_tens, self.timestep)
            #print(np.shape(pred))
            output[:,:,0] = torch.reshape(pred,(self.ensemble_size,self.input_size)) #output will now batch to be size (batch_size*ensembles x input_size)

            #print('first output stored\n','output shape',np.shape(output))
            #self.myNet,torch.reshape(input_test_torch_batch[:,0,:],(num_batch,input_size,1))
            #print('shape of x',np.shape(x),'\n shape of  output',np.shape(output))

            for step in range(self.time_chunk_size-1): #run for however many time steps are specified by time_chunk_size
                pred = self.step_method(self.myNet,torch.reshape(output[:,:,step],(self.ensemble_size*self.batch_size, self.input_size,1)),self.timestep)
                output[:,:,step+1] = torch.reshape(pred, (self.ensemble_size,self.input_size))
                #print('iterative output stored')
            #print('size output after iteration',np.shape(output))
        output = torch.reshape(output,(self.batch_size, self.ensemble_size, self.input_size, self.time_chunk_size)) #return back to batchsize x ensembles x inputsize x timechunksize
        #print('size output before kalman',np.size(output))
        output[:,:,:,-1] = self.EnsKF_step_module(output[:,:,:,-1]) #takes last prediction and corrects it ALSO: has gradients stored
        return output #batch size x ensem x inputsize x timchunksize

    def loss_function(self, Ens_ten, true_state_t):
        #Ens_ten = torch.reshape(Ens_ten,(self.batch_size,self.ensemble_size,self.input_size)) 
        #^^takes ((ensembles*batches) x inputsize) and makes it (batches x ensembles x inputsize)
        #so that we can still do batch learning
        #print('Shape Ens_ten:',np.shape(Ens_ten),'\nShape true_state_t:',np.shape(true_state_t))
        
        # avg_ens = torch.mean(Ens_ten,dim=1) #means along the ensemble size
        # # print('Shape avg_ens:',np.shape(avg_ens))
        # loss = torch.sqrt(torch.mean((true_state_t-avg_ens)**2, 1)) #means along spacial dimension, retains batch size => output is vector 1xbatchsize
        # # print('loss shape:',np.shape(loss))
        # # print(true_state_t.shape, Ens_ten.shape)
        
        loss = torch.sqrt(((true_state_t.unsqueeze(1) - Ens_ten)**2).mean(dim=1))

        return loss