import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, imgChannels, out_channels, num_filters, dimx, dimy, zDim):
        super(VAE, self).__init__()
        self.num_filters = num_filters
        self.dimx = dimx
        self.dimy = dimy
        self.featureDim = num_filters*dimx*dimy

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = (nn.Conv1d(imgChannels, self.num_filters, kernel_size=5, stride=1, padding='same'))
        self.encConv2 = (nn.Conv1d(self.num_filters, self.num_filters, kernel_size=5, stride=1, padding='same'))
        self.encConv3 = (nn.Conv1d(self.num_filters, self.num_filters, kernel_size=5, stride=1, padding='same'))
        self.encConv4 = (nn.Conv1d(self.num_filters, self.num_filters, kernel_size=5, stride=1, padding='same'))
        self.encFC1 = nn.Linear(self.featureDim, zDim)
        self.encFC2 = nn.Linear(self.featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, self.featureDim)
        self.decConv1 = (nn.Conv1d(self.num_filters, self.num_filters, kernel_size=5, stride=1, padding='same' ))
        self.decConv2 = (nn.Conv1d(self.num_filters, self.num_filters, kernel_size=5, stride=1, padding='same' ))
        self.decConv3 = (nn.Conv1d(self.num_filters, self.num_filters, kernel_size=5, stride=1, padding='same' ))
        self.decConv4 = (nn.Conv1d(self.num_filters, out_channels, kernel_size=5, stride=1, padding='same' ))
        

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = x.reshape(-1, 1, self.dimx)
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = x.view(-1, self.num_filters*self.dimx*self.dimy)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, self.num_filters, self.dimx)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = (self.decConv4(x)).reshape(-1, self.dimx)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

