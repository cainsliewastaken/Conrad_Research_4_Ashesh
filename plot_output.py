#%%
#run as jupyter notebook to plot
import matplotlib.pyplot as plt
import numpy as np
from canonicalPDEs import load_sol
u_truth_long = load_sol('KS_1024')


#%%
skip_factor = 100 # only plot the nth timestep, determined by value of skip_factor. Saves memory
plt.contourf(u_truth_long[:,::skip_factor])
temp, temp, y_min, y_max = plt.axis()
yrange = [-50, 0, 50]
plt.yticks([y_min, y_max/2, y_max] , yrange);
plt.xticks(ticks=plt.xticks()[0][1:], labels=1e-3 * skip_factor * np.array(plt.xticks()[0][1:], dtype=np.float64));

#%%
u_truth_long = u_truth_long[:,0:250000]

# %%
u_1d_f_spec_tdim = np.zeros(np.shape(u_truth_long.T[:-1,:]), dtype=complex)
for n in range(np.shape(u_truth_long)[1]-1):
    u_1d_f_spec_tdim[n,:] = np.abs(np.fft.fft(u_truth_long[:,n]))


#%%
plt.loglog(np.mean(u_1d_f_spec_tdim[:,1:512], axis=0))


# %%
