"""
@author: rmojgani
Model discovery
Run the KS model without u_xxxx, 
Find the u_xxxx using RVM
"""
import numpy as np
from canonicalPDEs import save_sol
from canonicalPDEs import ETRNK4intKS as intKS 
L = 100
N = 1024
dx = L/N
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

kappa = 2 * np.pi*np.fft.fftfreq(N,d=dx)

lambdas = [1.0,1.0,1.0]

u0 = -np.cos(x*2*np.pi/100)*(1+np.sin(-x*2*np.pi/100))


dt = 1e-3
Nt_spinoff = int(200.0/dt)
Nt = int(500.0/dt)
t = np.arange(0,Nt_spinoff*dt,dt) 
dt = t[1]-t[0]


print('starting spin up',Nt_spinoff)
u_spun_full = intKS(u0,t,kappa,N,lambdas)
u_spun = u_spun_full[:,-2] #solver leaves last timestep as zeros :(

save_sol(u_truth_long, 'KS_1024')

t = np.arange(Nt_spinoff*dt ,Nt*dt,dt) 
dt = t[1]-t[0]

print('/ Simulation start ... ')
print('dt',dt)
print('end time',Nt*dt)

u_truth_long = intKS(u_spun,t,kappa,N,lambdas)
#save_sol(u_truth_long, 'KS_1024')


