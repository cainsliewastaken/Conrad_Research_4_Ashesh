import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----
def RK4_solve(odefunc,TimeSpan,uhat0,nt):
    h = float(TimeSpan[1]-TimeSpan[0])/nt  
    print(h) 
    w = uhat0
    out_vals = np.zeros([nt, w.shape[0]])
    t = TimeSpan[0]
    for i in np.arange(nt):
        w = RK4Step(odefunc, t, w, h)
        t = t+h
        out_vals[i] = np.real(np.fft.ifft(vhat2uhat(t,w)))
    return out_vals

def RK4Step(odefunc, t,w,h):
    k1 = odefunc(t,w)
    k2 = odefunc(t+0.5*h, w+0.5*k1*h)
    k3 = odefunc(t+0.5*h, w+0.5*k2*h)
    k4 = odefunc(t+h,     w+k3*h)
    return w + (k1+2*k2+2*k3+k4)*(h/6.)

#----- Constructing the grid -----
L   = 2.
nx  = 512
x   = np.linspace(0.,L, nx+1)
x   = x[:nx]  

kx1 = np.linspace(0,nx/2-1, int(nx/2))
kx2 = np.linspace(1,nx/2,  int(nx/2))
kx2 = -1*kx2[::-1]
kx  = (2.* np.pi/L)*np.concatenate((kx1,kx2))

#------ Initial conditions -----
u0      = np.cos(np.pi*x)
uhat0   = np.fft.fft(u0)

#----- Parameters -----
delta  = 0.022
delta2 = delta**2
nt = 150000
t0 = 0
tf = nt*delta/10
# tf = 1./np.pi
TimeSpan = [t0, tf] 

#----- Change of Variables -----
def uhat2vhat(t,uhat):
    return np.exp( -1j * (kx**3) * delta2 * t) * uhat

def vhat2uhat(t,vhat):
    return np.exp(1j * (kx**3) * delta2 * t) * vhat

#----- Define RHS -----
def uhatprime(t, uhat):
    u = np.fft.ifft(uhat)
    return 1j * (kx**2) * delta2 * uhat - 0.5j * kx * np.fft.fft(u**2)

def vhatprime(t, vhat):
    u = np.fft.ifft(vhat2uhat(t,vhat))
    return  -0.5j * kx * uhat2vhat(t, np.fft.fft(u**2) )

#----- Initial condition -----
vhat0    = uhat2vhat(t0,uhat0)

file_path ='/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/'
file_name = 'KdV_data.npz'

#------ Solving for ODE -----

out_vals = RK4_solve(vhatprime,[t0,tf],vhat0,nt)

np.savez(file_path+file_name, out_vals)