import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, distribution_size):
        super(VAE, self).__init__()

        
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.encFC1 = nn.Linear(hidden_layer_size, distribution_size)
        self.encFC2 = nn.Linear(hidden_layer_size, distribution_size)
        self.endFC3 = nn.Linear(hidden_layer_size, distribution_size)

        
        self.decFC1 = nn.Linear(2*distribution_size, hidden_layer_size)
        self.declinear1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.declinear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.declinear3 = nn.Linear(hidden_layer_size, output_size)
        

    def encoder(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        x = self.endFC3(x)
        return x, mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        
        x = F.tanh(self.decFC1(z))
        x = F.tanh(self.declinear1(x))
        x = F.tanh(self.declinear2(x))
        x = self.declinear3(x)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        x, mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(torch.cat([x,z], dim=-1))
        return out, mu, logVar

