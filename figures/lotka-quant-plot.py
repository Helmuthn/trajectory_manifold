import matplotlib.pyplot as plt
import numpy as np

###############################
##### Read in Data Arrays #####
###############################

data = np.load("quant_noise_lotka.npz")

noise_power   = data['noise_power']

mae_MMSE      = data['mae_MMSE']
mae_MAP       = data['mae_MAP']
mae_ML        = data['mae_ML']
mae_MMSE_proj = data['mae_MMSE_proj']
mae_MMSE_init = data['mae_MMSE_init']

mse_MMSE      = data['mse_MMSE']
mse_MAP       = data['mse_MAP']
mse_ML        = data['mse_ML']
mse_MMSE_proj = data['mse_MMSE_proj']
mse_MMSE_init = data['mse_MMSE_init']

sup_MMSE      = data['sup_MMSE']
sup_MAP       = data['sup_MAP']
sup_ML        = data['sup_ML']
sup_MMSE_proj = data['sup_MMSE_proj']
sup_MMSE_init = data['sup_MMSE_init']


data = np.load("quant_horizon_lotka.npz")

time_horizons   = data['time_horizons']

t_mae_MMSE      = data['t_mae_MMSE']
t_mae_MAP       = data['t_mae_MAP']
t_mae_ML        = data['t_mae_ML']
t_mae_MMSE_proj = data['t_mae_MMSE_proj']
t_mae_MMSE_init = data['t_mae_MMSE_init']

t_mse_MMSE      = data['t_mse_MMSE']
t_mse_MAP       = data['t_mse_MAP']
t_mse_ML        = data['t_mse_ML']
t_mse_MMSE_proj = data['t_mse_MMSE_proj']
t_mse_MMSE_init = data['t_mse_MMSE_init']

t_sup_MMSE      = data['t_sup_MMSE']
t_sup_MAP       = data['t_sup_MAP']
t_sup_ML        = data['t_sup_ML']
t_sup_MMSE_proj = data['t_sup_MMSE_proj']
t_sup_MMSE_init = data['t_sup_MMSE_init']


#####################################
##### Configure Plotting Fonts  #####
#####################################

SMALL_SIZE = 11
MEDIUM_SIZE = SMALL_SIZE
BIGGER_SIZE = 11
plt.rc('font', size=SMALL_SIZE, family="DejaVu Serif")          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


##############################################
##### Plot As A Function Of Noise Power  #####
##############################################

fig, axs = plt.subplots(2,2, sharex=True)

fig.set_figwidth(5.5)
fig.set_figheight(4)
axs[1,0].plot(noise_power, mae_MMSE, label="MMSE Trajectory, Ambient")
axs[1,0].plot(noise_power, mae_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,0].plot(noise_power, mae_MAP, label="MAP Trajectory")
axs[1,0].plot(noise_power, mae_ML, label="ML/MAP Initial Condition")
axs[1,0].plot(noise_power, mae_MMSE_init, label="MMSE Initial Condition")
axs[1,0].set_xscale('log')
axs[1,0].set_title("Mean Absolute Error")
axs[1,0].set_xlim(0.01,10)

axs[1,1].plot(noise_power, mse_MMSE, label="MMSE Trajectory, Ambient")
axs[1,1].plot(noise_power, mse_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,1].plot(noise_power, mse_MAP, label="MAP Trajectory")
axs[1,1].plot(noise_power, mse_ML, label="ML/MAP Initial Condition")
axs[1,1].plot(noise_power, mse_MMSE_init, label="MMSE Initial Condition")
axs[1,1].set_xscale('log')
axs[1,1].set_title("Mean Squared Error")
axs[1,1].set_xlim(0.01,10)

axs[0,0].plot(noise_power, sup_MMSE, label="MMSE Trajectory, Ambient")
axs[0,0].plot(noise_power, sup_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[0,0].plot(noise_power, sup_MAP, label="MAP Trajectory")
axs[0,0].plot(noise_power, sup_ML, label="ML/MAP Initial Condition")
axs[0,0].plot(noise_power, sup_MMSE_init, label="MMSE Initial Condition")
axs[0,0].set_xscale('log')
axs[0,0].set_title("Sup Norm")
axs[0,0].set_xlim(0.01,10)

axs[0,1].axis("off")

fig.text(0.5, 0, 'Noise Power', ha='center')
fig.text(0.0, 0.5, 'Error', va='center', rotation='vertical')

handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,1].legend(handles, labels, loc='center')

for i in range(2):
    for j in range(2):
        axs[i,j].grid(True, which='major')

fig.tight_layout()
fig.savefig("quantitative_snr.pdf", format="pdf", bbox_inches="tight")


##############################################
##### Plot As A Function Of Time Horizon #####
##############################################

fig, axs = plt.subplots(2,2, sharex=True)

fig.set_figwidth(5.5)
fig.set_figheight(4)
axs[1,0].plot(time_horizons, t_mae_MMSE, label="MMSE Trajectory, Ambient")
axs[1,0].plot(time_horizons, t_mae_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,0].plot(time_horizons, t_mae_MAP, label="MAP Trajectory")
axs[1,0].plot(time_horizons, t_mae_ML, label="ML/MAP Initial Condition")
axs[1,0].plot(time_horizons, t_mae_MMSE_init, label="MMSE Initial Condition")
axs[1,0].set_title("Mean Absolute Error")
axs[1,0].set_xlim(2,30)
axs[1,0].set_ylim(0,30)

axs[1,1].plot(time_horizons, t_mse_MMSE, label="MMSE Trajectory, Ambient")
axs[1,1].plot(time_horizons, t_mse_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,1].plot(time_horizons, t_mse_MAP, label="MAP Trajectory")
axs[1,1].plot(time_horizons, t_mse_ML, label="ML/MAP Initial Condition")
axs[1,1].plot(time_horizons, t_mse_MMSE_init, label="MMSE Initial Condition")
axs[1,1].set_title("Mean Squared Error")
axs[1,1].set_xlim(2,30)
axs[1,1].set_ylim(0,30)

axs[0,0].plot(time_horizons, t_sup_MMSE, label="MMSE Trajectory, Ambient")
axs[0,0].plot(time_horizons, t_sup_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[0,0].plot(time_horizons, t_sup_MAP, label="MAP Trajectory")
axs[0,0].plot(time_horizons, t_sup_ML, label="ML/MAP Initial Condition")
axs[0,0].plot(time_horizons, t_sup_MMSE_init, label="MMSE Initial Condition")
axs[0,0].set_title("Sup Norm")
axs[0,0].set_xlim(2,30)

axs[0,1].axis("off")

fig.text(0.5, 0, 'Time Horizon', ha='center')
fig.text(0.0, 0.5, 'Error', va='center', rotation='vertical')

handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,1].legend(handles, labels, loc='center')

for i in range(2):
    for j in range(2):
        axs[i,j].grid(True, which='major')

fig.tight_layout()

fig.savefig("quantitative_horizon.pdf", format="pdf", bbox_inches="tight")