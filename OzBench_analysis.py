#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ### OzBench Cartesian Models Analysis

# %%
import numpy as np
from matplotlib import pyplot as plt
import h5py

# %% [markdown]
# #### Scaling of parameters

# %%
rho_M             = 1.
g_M               = 1.
Height_M          = 1.
viscosity_M       = 1.


# %%
rho_N             = 80.0  # kg/m**3  note delta rho
g_N               = 9.81    # m/s**2
Height_N          = 1000e3   # m
viscosity_N       = 2e19  #Pa.sec or kg/m.sec


# %%
#Non-dimensional (scaling)
rho_scaling 		= rho_N/rho_M
viscosity_scaling 	= viscosity_N/viscosity_M
g_scaling 			= g_N/g_M
Height_scaling 		= Height_N/Height_M
pressure_scaling 	= rho_scaling * g_scaling * Height_scaling
pressure_scaling_mpa= pressure_scaling/1e6
time_scaling 		= viscosity_scaling/pressure_scaling
strainrate_scaling 	= 1./time_scaling
mm_yr               = 1e3*365*24*60*60
velocity_scaling    = (Height_scaling/time_scaling)*mm_yr

# %% [markdown]
# \begin{align}
# {\tau}_N = \frac{{\rho}_{0N}{g}_N{l}_N}{{\rho}_{0M}{g}_M{l}_M} {\tau}_M
# \end{align}
#
# \begin{align}
# {V}_N = \frac{{\eta}_{0M}}{{\eta}_{0N}}\frac{{\rho}_{0N}{g}_N{{l}_N}^2}{{\rho}_{0M}{g}_M{{l}_M}^2} {V}_M
# \end{align}

# %%
# loading files
uw2_path = '/Users/tgol0003/uw_thyagi/uw_codes/OzBench_dim/'
uw3_path = '/Users/tgol0003/uw3_folder/underworld3/Jupyterbook/Notebooks/Examples-SegmentOfAnnulus-Sphere/output/'

uw2_dir = uw2_path+'output/'
uw3_sm_dir = uw3_path+'OzBench_Cart_Mod_BC_SMesh_Test1/'
uw3_usm_dir = uw3_path+'OzBench_Cart_Mod_BC_USMesh_Test1/'

# %%
# reading uw2 files
pfs_data_uw2 = h5py.File(uw2_dir+'profiles_data.h5')

# reading uw3 files from structured mesh
pfs_data_uw3_sm = h5py.File(uw3_sm_dir+'profiles_data.h5')

# reading uw3 files from unstructured mesh
pfs_data_uw3_usm = h5py.File(uw3_usm_dir+'profiles_data.h5')

# %%
# plotting velocities

plt.figure(dpi=150)

pfs_data_list = [pfs_data_uw2, pfs_data_uw3_sm, pfs_data_uw3_usm]
depth_list = [15, 50, 85]
step_list = [2] # [0, 1, 2]


linestyle_list = ['-', '--', '-.']
color_list = ['C0', 'C1', 'C2']
label_list = ['(UW2)', '(UW3_SM)', '(UW3_USM)']

for k, pf_data in enumerate(pfs_data_list):
    for j, depth in enumerate(depth_list):
        for i, step in enumerate(step_list):
            pf_name = pf_data[str(depth)+'/'+str(step)]
            plt.plot(pf_name['coords'][()][:,0], 
                     np.sqrt(pf_name['velocity'][()][:,0]**2 + pf_name['velocity'][()][:,1]**2)*velocity_scaling, 
                     label=str(depth)+' km '+label_list[k], linewidth=1, color=color_list[j], 
                     linestyle=linestyle_list[k])


plt.xlabel('X coord')
plt.ylabel(r'Velocity (mm/yr)')
plt.title('')
plt.grid()
plt.xlim(0.8, 1.5)
plt.ylim(0, 500)
plt.legend(loc='upper right', fontsize=8, ncol=1)

# %%
# plotting strain rate inv

plt.figure(dpi=150)

for k, pf_data in enumerate(pfs_data_list):
    for j, depth in enumerate(depth_list):
        for i, step in enumerate(step_list):
            pf_name = pf_data[str(depth)+'/'+str(step)]
            plt.plot(pf_name['coords'][()][:,0], np.log10(pf_name['strainRateInv'][()]*strainrate_scaling), 
                     label=str(depth)+' km '+label_list[k], linewidth=1, color=color_list[j], 
                     linestyle=linestyle_list[k])


plt.xlabel('X coord')
plt.ylabel(r'Strain Rate $2^{nd}$ Invariant (1/s)')
plt.title('')
plt.grid()
plt.xlim(0.8, 1.5)
plt.ylim(-15.75, -12.5)
plt.legend(loc='upper right', fontsize=8, ncol=1)

# %%
# pfs_data_uw3_usm['85/2']['strainRateInv'][()]

# %%
# # plotting viscosity

plt.figure(dpi=150)

for k, pf_data in enumerate(pfs_data_list):
    for j, depth in enumerate(depth_list):
        for i, step in enumerate(step_list):
            pf_name = pf_data[str(depth)+'/'+str(step)]
            plt.plot(pf_name['coords'][()][:,0], pf_name['viscosity'][()]*viscosity_scaling, 
                     label=str(depth)+' km '+label_list[k], linewidth=1, color=color_list[j], 
                     linestyle=linestyle_list[k])
            

plt.xlabel('X coord')
plt.ylabel(r'Viscosity')
plt.title('')
plt.grid()
plt.xlim(0.8, 1.5)
plt.ylim(-1.1e21, 1.1e22)
plt.legend(loc='lower right', fontsize=8, ncol=1)

plt.savefig('./viscosity.png')

# %%
# pfs_data_uw3_usm['85/2']['viscosity'][()]

# %%
# plotting stress invariant

plt.figure(dpi=150)

for k, pf_data in enumerate(pfs_data_list):
    for j, depth in enumerate(depth_list):
        for i, step in enumerate(step_list):
            pf_name = pf_data[str(depth)+'/'+str(step)]
            plt.plot(pf_name['coords'][()][:,0], pf_name['stressInv'][()]*pressure_scaling_mpa, 
                     label=str(depth)+' km '+label_list[k], linewidth=1, color=color_list[j], 
                     linestyle=linestyle_list[k])
            

plt.xlabel('X coord')
plt.ylabel(r'Stress 2nd Invariant (MPa)')
plt.title('')
plt.grid()
plt.xlim(0.8, 1.5)
plt.ylim(-10, 800)
plt.legend(loc='upper right', fontsize=8, ncol=1)


# %%
