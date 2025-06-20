
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----

def RK4Step(u,kx):
    k1 = rhs(u,kx)
    k2 = rhs(u+0.5*dt*k1,kx)
    k3 = rhs(u+0.5*dt*k2,kx)
    k4 = rhs(u+dt*k3,kx)
    return u + (k1+2*k2+2*k3+k4)*(dt/6.)

#----- Constructing the grid -----
L = 2*np.pi
# L = 20
m = 16
x = np.arange(-m/2,m/2)*(L/m)
print('X min and max',x.min(), x.max(),', dx ', x[1]-x[0])
kx = np.fft.fftfreq(m)*m*2*np.pi/L

#----- Parameters -----
alpha = 5
# delta  = 0.022*150
# delta  = 0.5
# delta3 = delta**3
delta3 = 1
# nu = 0
nu = 0.1

#----- Define RHS -----
def rhs(uhat, kx):
    u = np.fft.ifft(uhat)
    return alpha*np.fft.fft(u*(np.fft.ifft(1j*kx*uhat))) - delta3*(-1j*kx**3*uhat) -nu*kx**2*uhat

file_name = 'KdV_data_spectral_alpha_'+str(alpha)+'_delta_'+str(delta3)+'_nu_'+str(nu)+'_long'
file_path ='/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/'+file_name+'/'
print(file_name)

# prev_chunk = np.load('/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KdV_data_spectral_delta_1.1_nu_0.1_long_v2/KdV_data_spectral_delta_1.1_nu_0.1_long_v2_chunk_99.npz')
# last_dt = prev_chunk['arr_0'][:,-1]
#------ Initial conditions -----
u0      = (0.15)*np.sin(x)#*np.exp(-0.2*x**2)
# u0 = last_dt
uhat0   = np.fft.fft(u0)
print('u min and max',u0.min(), u0.max())

#------ Solving for ODE -----
t0 = 0
# dt = 1.73/((m/2)**3)
dt = 1e-3

nt = 2000000

usim = np.zeros([m,nt])
# print(usim.shape)

for i in range (0, nt):
  if i == 0:
    upred = uhat0
    usim [:,i] = np.real(np.fft.ifft(upred))
  else:
    upred = RK4Step(upred,kx)
    usim [:,i] = np.real(np.fft.ifft(upred))
  if i%100000==0:
    print(i)
    print((np.abs(usim[:,i]).max()*dt)/np.abs((x[1]-x[0])), np.max(usim[:,i]), np.min(usim[:,i]))
  if np.isnan(upred).any():
     print('Nan found at timestep '+str(i))
     break
print('Output finished')

prev_ind = 0
chunk_count = 0
num_chunks = 100
print('Starting save')
for chunk in np.array_split(usim.T, num_chunks):
    current_ind = prev_ind + chunk.shape[0]
    np.savez(file_path+file_name+'_chunk_'+str(chunk_count)+'.npz', usim[:, prev_ind:current_ind])
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')

plt.contourf(usim[:,::1000])
plt.savefig(file_name+'.png')
print('Figure saved')