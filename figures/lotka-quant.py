#%%
import trajectory_manifold.examples as examples
import trajectory_manifold.estimation as estimation
from trajectory_manifold.manifold import SolverParameters

from diffrax import ODETerm, SaveAt, PIDController, diffeqsolve, Heun

import jax.numpy as jnp
from jax import random, jit, vmap

from math import pi

from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

## Setup Problem Specification

vector_field = examples.lotka_volterra_vector_field(1,2,4,2)
noise_std = 1 # Noise Power
center = 1.2 # Center of sample grid
delta = 0.05 # Grid Step Size

parameters = SolverParameters(relative_tolerance = 1e-2,
                              absolute_tolerance = 1e-2,
                              step_size = 0.1,
                              time_interval = (0,10),
                              solver=Heun(),
                              max_steps=16**5)

## Read In Trajectory Weight Grid

print("Reading in weight matrix")

npzfile = jnp.load("lotkaweights.npz")
samples = npzfile['samples']
x = npzfile['x']
y = npzfile['y']
weight_matrix = npzfile['weight_matrix']
weight_matrix = jnp.array(weight_matrix)
weight_matrix = jnp.reshape(weight_matrix, (x.shape[0], y.shape[0]))

#%%
subsample = 3

def max_estimate(grid):
    peak = jnp.argmax(grid)
    peak_x = x[peak % x.shape[0]]
    peak_y = y[peak // y.shape[0]]
    return jnp.stack([peak_x, peak_y])

def min_estimate(grid):
    peak = jnp.argmin(grid)
    peak_x = x[peak % x.shape[0]]
    peak_y = y[peak // y.shape[0]]
    return jnp.stack([peak_x, peak_y])



def state_log_prior(state):
    return -1 * jnp.log(9)


def construct_log_likelihood_grid(observation_times, observations):
    log_likelihood = estimation.trajectory_log_likelihood(vector_field,
                                                          observations,
                                                          observation_times, 
                                                          observation_log_likelihood,
                                                          parameters)
                        
    v_ll = vmap(log_likelihood)
    grid_ll = v_ll(samples.T)
    grid_ll = jnp.reshape(grid_ll, (x.shape[0], y.shape[0]))
    return grid_ll
    
@jit
def construct_observations(std, key, parameters):
    key, subkey = random.split(key)
    true_init = 2 * random.uniform(subkey, shape=(2,)) + center - 1

    term = ODETerm(vector_field)
    solver = parameters.solver
    observation_times = jnp.arange(parameters.time_interval[0], 
                                   parameters.time_interval[1] + parameters.step_size, 
                                   step=parameters.step_size)

    saveat = SaveAt(ts = observation_times)

    stepsize_controller = PIDController(rtol = parameters.relative_tolerance,
                                        atol = parameters.absolute_tolerance)

    states = diffeqsolve(term,
                         solver,
                         t0 = parameters.time_interval[0],
                         t1 = parameters.time_interval[1],
                         dt0 = parameters.step_size,
                         saveat = saveat,
                         stepsize_controller = stepsize_controller,
                         y0 = true_init).ys


    key, subkey = random.split(key)
    noise = std*random.normal(subkey, shape=states.shape)
    observations = states + noise
    observation_times = observation_times[:31]
    observations = observations[:31,:]
    return true_init, observation_times[::subsample], observations[::subsample,:]


@jit
def SolveODE(initial_state):
    term = ODETerm(vector_field)
    observation_times = jnp.arange(parameters.time_interval[0], 
                                   parameters.time_interval[1] + parameters.step_size, 
                                   step=parameters.step_size)

    saveat = SaveAt(ts = observation_times)
    stepsize_controller = PIDController(rtol = parameters.relative_tolerance,
                                        atol = parameters.absolute_tolerance)
    return diffeqsolve(term,
                       solver = parameters.solver,
                       t0 = parameters.time_interval[0],
                       t1 = parameters.time_interval[1],
                       dt0 = 0.1,
                       saveat = saveat,
                       stepsize_controller = stepsize_controller,
                       y0 = initial_state).ys


def square_distance_tensor(solution_set):
    tensor_shape = (solution_set.shape[0],
                    solution_set.shape[1],
                    solution_set.shape[0],
                    solution_set.shape[1])
    out = jnp.zeros(tensor_shape)
    for i in range(solution_set.shape[0]):
        for j in range(solution_set.shape[1]):
            solution = solution_set[i,j,:,:]
            distance_mat = jnp.sum(jnp.square(solution[None, None, :, :] - solution_set), axis=(2,3)) * parameters.step_size
            out = out.at[i,j,:,:].set(distance_mat)
    return out

@jit
def mse_with_distance(density, distance_mat):
    out = jnp.zeros_like(density)
    for i in range(distance_mat.shape[2]):
        for j in range(distance_mat.shape[3]):
            out = out.at[i,j].set(jnp.sum(density[None, None, :, :] * distance_mat[:,:,i,j]))
    return out*(delta**2)


solveODE_v = vmap(SolveODE)
solution_set = solveODE_v(samples.T)
solution_set = jnp.reshape(solution_set, (x.shape[0], x.shape[0],101,2))
distance_mat = square_distance_tensor(solution_set)

sample_grid = jnp.reshape(samples.T, (x.shape[0], y.shape[0], 2))

# %%
# Vary the noise power, compute MSE and MAE
# Step noise power in log scale
noise_power = jnp.power(10, jnp.arange(-2,2.2,0.2))

key = random.PRNGKey(123)

mse_MMSE = [0] * noise_power.shape[0]
mae_MMSE = [0] * noise_power.shape[0]
sup_MMSE = [0] * noise_power.shape[0]

mse_MMSE_proj = [0] * noise_power.shape[0]
mae_MMSE_proj = [0] * noise_power.shape[0]
sup_MMSE_proj = [0] * noise_power.shape[0]

mse_MMSE_init = [0] * noise_power.shape[0]
mae_MMSE_init = [0] * noise_power.shape[0]
sup_MMSE_init = [0] * noise_power.shape[0]

mse_MAP  = [0] * noise_power.shape[0]
mae_MAP  = [0] * noise_power.shape[0]
sup_MAP  = [0] * noise_power.shape[0]

mse_ML   = [0] * noise_power.shape[0]
mae_ML   = [0] * noise_power.shape[0]
sup_ML   = [0] * noise_power.shape[0]

mcmc_samples = 10000

outer_bar = tqdm(range(len(noise_power)), desc="Noise Step")
inner_bar = tqdm(range(mcmc_samples), leave=False, desc="    MC Step")

for i, variance in enumerate(noise_power):
    inner_bar.reset()
    std = jnp.sqrt(variance)

    @jit
    def observation_log_likelihood(observation, state):
        noise_std = std
        partition = jnp.power(2 * pi, -observation.shape[0]/2.0)
        return jnp.log(partition) -1 * jnp.sum(jnp.square(observation - state))/(2*noise_std**2)

    for j in range(mcmc_samples):
        # Generate Data
        key, subkey = random.split(key)
        true_init, observation_times, observations = construct_observations(std, subkey, parameters)
        true_solution = SolveODE(true_init)

        # Constuct Distribution Grids
        grid_ll = construct_log_likelihood_grid(observation_times, observations)
        grid_pt = grid_ll - jnp.log(weight_matrix)
        grid_ps = jnp.exp(grid_ll - jnp.max(grid_ll))
        grid_ps = grid_ps / jnp.sum(grid_ps)

        # Find maximum of grid for ML + MAP
        ml_est = max_estimate(grid_ll)
        map_state_est = ml_est
        map_trajectory_est = max_estimate(grid_pt)

        ml_trajectory = SolveODE(ml_est)
        map_trajectory = SolveODE(map_trajectory_est)

        # Compute MMSE initial condition
        mmse_initial_cond = sample_grid * grid_ps[:,:,None]
        mmse_initial_cond = jnp.sum(mmse_initial_cond, axis=(0,1))
        mmse_traject_init = SolveODE(mmse_initial_cond)

        # Construct MMSE trajectory
        mmse_trajectory_est = solution_set * grid_ps[:,:, None, None]
        mmse_trajectory_est = jnp.sum(mmse_trajectory_est, axis=(0,1))

        # Construct MMSE trajectory on manifold
        mse_mat = mse_with_distance(grid_ps, distance_mat)
        mmse_init = min_estimate(mse_mat)
        mmse_trajectory_proj = SolveODE(mmse_init)

        # Compute Error
        mse_MMSE[i] += jnp.sum(jnp.square(mmse_trajectory_est - true_solution))
        mae_MMSE[i] += jnp.sum(jnp.abs(mmse_trajectory_est - true_solution))
        sup_MMSE[i] += jnp.max(jnp.abs(mmse_trajectory_est - true_solution))

        mse_MMSE_proj[i] += jnp.sum(jnp.square(mmse_trajectory_proj - true_solution))
        mae_MMSE_proj[i] += jnp.sum(jnp.abs(mmse_trajectory_proj - true_solution))
        sup_MMSE_proj[i] += jnp.max(jnp.abs(mmse_trajectory_proj - true_solution))

        mse_MMSE_init[i] += jnp.sum(jnp.square(mmse_traject_init - true_solution))
        mae_MMSE_init[i] += jnp.sum(jnp.abs(mmse_traject_init - true_solution))
        sup_MMSE_init[i] += jnp.max(jnp.abs(mmse_traject_init - true_solution))

        mse_ML[i] += jnp.sum(jnp.square(ml_trajectory - true_solution))
        mae_ML[i] += jnp.sum(jnp.abs(ml_trajectory - true_solution))
        sup_ML[i] += jnp.max(jnp.abs(ml_trajectory - true_solution))

        mse_MAP[i] += jnp.sum(jnp.square(map_trajectory - true_solution))
        mae_MAP[i] += jnp.sum(jnp.abs(map_trajectory - true_solution))
        sup_MAP[i] += jnp.max(jnp.abs(map_trajectory - true_solution))

        # Update Progress Bar
        inner_bar.update(1)

    mse_MMSE[i] /= mcmc_samples / parameters.step_size 
    mae_MMSE[i] /= mcmc_samples / parameters.step_size 
    sup_MMSE[i] /= mcmc_samples 
    mse_MMSE_proj[i] /= mcmc_samples / parameters.step_size 
    mae_MMSE_proj[i] /= mcmc_samples / parameters.step_size 
    sup_MMSE_proj[i] /= mcmc_samples 
    mse_MMSE_init[i] /= mcmc_samples / parameters.step_size 
    mae_MMSE_init[i] /= mcmc_samples / parameters.step_size 
    sup_MMSE_init[i] /= mcmc_samples 
    mse_ML[i]   /= mcmc_samples / parameters.step_size 
    mae_ML[i]   /= mcmc_samples / parameters.step_size 
    sup_ML[i]   /= mcmc_samples 
    mse_MAP[i]  /= mcmc_samples / parameters.step_size 
    mae_MAP[i]  /= mcmc_samples / parameters.step_size 
    sup_MAP[i]  /= mcmc_samples 
    outer_bar.update(1)

    

    
# %%
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

fig, axs = plt.subplots(2,2, sharex=True)

fig.set_figwidth(5.5)
fig.set_figheight(4)
axs[1,0].plot(noise_power, mae_MMSE, label="MMSE Trajectory, Ambient")
axs[1,0].plot(noise_power, mae_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,0].plot(noise_power, mae_MAP, label="MAP Trajectory")
axs[1,0].plot(noise_power, mae_ML, label="ML/MAP Initial Condition")
axs[1,0].plot(noise_power, mae_MMSE_init, label="MMSE Initial Condition")
axs[1,0].set_xscale('log')
#axs[1,0].set_yscale('log')
axs[1,0].set_title("Mean Absolute Error")
axs[1,0].set_xlim(0.01,10)

axs[1,1].plot(noise_power, mse_MMSE, label="MMSE Trajectory, Ambient")
axs[1,1].plot(noise_power, mse_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,1].plot(noise_power, mse_MAP, label="MAP Trajectory")
axs[1,1].plot(noise_power, mse_ML, label="ML/MAP Initial Condition")
axs[1,1].plot(noise_power, mse_MMSE_init, label="MMSE Initial Condition")
axs[1,1].set_xscale('log')
#axs[1,1].set_yscale('log')
axs[1,1].set_title("Mean Squared Error")
axs[1,1].set_xlim(0.01,10)

axs[0,0].plot(noise_power, sup_MMSE, label="MMSE Trajectory, Ambient")
axs[0,0].plot(noise_power, sup_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[0,0].plot(noise_power, sup_MAP, label="MAP Trajectory")
axs[0,0].plot(noise_power, sup_ML, label="ML/MAP Initial Condition")
axs[0,0].plot(noise_power, sup_MMSE_init, label="MMSE Initial Condition")
axs[0,0].set_xscale('log')
#axs[0,0].set_yscale('log')
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
fig.show()

# %%
fig.savefig("quantitative_snr.pdf", format="pdf", bbox_inches="tight")

#%%
jnp.savez("quant_noise_lotka.npz", noise_power=noise_power,
                                    mse_MMSE = jnp.array(mse_MMSE),
                                    mae_MMSE = jnp.array(mae_MMSE),
                                    sup_MMSE = jnp.array(sup_MMSE),
                                    mse_MMSE_proj = jnp.array(mse_MMSE_proj),
                                    mae_MMSE_proj = jnp.array(mae_MMSE_proj),
                                    sup_MMSE_proj = jnp.array(sup_MMSE_proj),
                                    mse_MMSE_init = jnp.array(mse_MMSE_init),
                                    mae_MMSE_init = jnp.array(mae_MMSE_init),
                                    sup_MMSE_init = jnp.array(sup_MMSE_init),
                                    mse_ML = jnp.array(mse_ML),
                                    mae_ML = jnp.array(mae_ML),
                                    sup_ML = jnp.array(sup_ML),
                                    mse_MAP = jnp.array(mse_MAP),
                                    mae_MAP = jnp.array(mae_MAP),
                                    sup_MAP = jnp.array(sup_MAP))

# %%

# Vary the noise power, compute MSE and MAE
# Step noise power in log scale
time_horizons = jnp.arange(2,32,2)

key = random.PRNGKey(123)

t_mse_MMSE = [0] * time_horizons.shape[0]
t_mae_MMSE = [0] * time_horizons.shape[0]
t_sup_MMSE = [0] * time_horizons.shape[0]

t_mse_MMSE_proj = [0] * time_horizons.shape[0]
t_mae_MMSE_proj = [0] * time_horizons.shape[0]
t_sup_MMSE_proj = [0] * time_horizons.shape[0]

t_mse_MMSE_init = [0] * time_horizons.shape[0]
t_mae_MMSE_init = [0] * time_horizons.shape[0]
t_sup_MMSE_init = [0] * time_horizons.shape[0]

t_mse_MAP  = [0] * time_horizons.shape[0]
t_mae_MAP  = [0] * time_horizons.shape[0]
t_sup_MAP  = [0] * time_horizons.shape[0]

t_mse_ML   = [0] * time_horizons.shape[0]
t_mae_ML   = [0] * time_horizons.shape[0]
t_sup_ML   = [0] * time_horizons.shape[0]

mcmc_samples = 5000

outer_bar = tqdm(range(len(noise_power)), desc="Noise Step")
inner_bar = tqdm(range(mcmc_samples), leave=False, desc="    MC Step")

for i, horizon in enumerate(time_horizons):
    inner_bar.reset()
    std = 1
    parameters = SolverParameters(relative_tolerance = 1e-2,
                                  absolute_tolerance = 1e-2,
                                  step_size = 0.1,
                                  time_interval = (0,horizon),
                                  solver=Heun(),
                                  max_steps=16**5)

    @jit
    def observation_log_likelihood(observation, state):
        noise_std = std
        partition = jnp.power(2 * pi, -observation.shape[0]/2.0)
        return jnp.log(partition) -1 * jnp.sum(jnp.square(observation - state))/(2*noise_std**2)

    def construct_observations(std, key, parameters):
        key, subkey = random.split(key)
        true_init = 2 * random.uniform(subkey, shape=(2,)) + center - 1

        term = ODETerm(vector_field)
        solver = parameters.solver
        observation_times = jnp.arange(parameters.time_interval[0], 
                                       parameters.time_interval[1] + parameters.step_size, 
                                       step=parameters.step_size)

        saveat = SaveAt(ts = observation_times)

        stepsize_controller = PIDController(rtol = parameters.relative_tolerance,
                                            atol = parameters.absolute_tolerance)

        states = diffeqsolve(term,
                             solver,
                             t0 = parameters.time_interval[0],
                             t1 = parameters.time_interval[1],
                             dt0 = parameters.step_size,
                             saveat = saveat,
                             stepsize_controller = stepsize_controller,
                             y0 = true_init).ys


        key, subkey = random.split(key)
        noise = std*random.normal(subkey, shape=states.shape)
        observations = states + noise
        observation_times = observation_times[:31]
        observations = observations[:31,:]
        return true_init, observation_times[::subsample], observations[::subsample,:]


    def SolveODE(initial_state):
        term = ODETerm(vector_field)
        observation_times = jnp.arange(parameters.time_interval[0], 
                                       parameters.time_interval[1] + parameters.step_size, 
                                       step=parameters.step_size)

        saveat = SaveAt(ts = observation_times)
        stepsize_controller = PIDController(rtol = parameters.relative_tolerance,
                                            atol = parameters.absolute_tolerance)
        return diffeqsolve(term,
                           solver = parameters.solver,
                           t0 = parameters.time_interval[0],
                           t1 = parameters.time_interval[1],
                           dt0 = 0.1,
                           saveat = saveat,
                           stepsize_controller = stepsize_controller,
                           y0 = initial_state).ys


    def square_distance_tensor(solution_set):
        tensor_shape = (solution_set.shape[0],
                        solution_set.shape[1],
                        solution_set.shape[0],
                        solution_set.shape[1])
        out = jnp.zeros(tensor_shape)
        for i in range(solution_set.shape[0]):
            for j in range(solution_set.shape[1]):
                solution = solution_set[i,j,:,:]
                distance_mat = jnp.sum(jnp.square(solution[None, None, :, :] - solution_set), axis=(2,3)) * parameters.step_size
                out = out.at[i,j,:,:].set(distance_mat)
        return out

    @jit
    def mse_with_distance(density, distance_mat):
        out = jnp.zeros_like(density)
        for i in range(distance_mat.shape[2]):
            for j in range(distance_mat.shape[3]):
                out = out.at[i,j].set(jnp.sum(density[None, None, :, :] * distance_mat[:,:,i,j]))
        return out*(delta**2)


    solveODE_v = vmap(SolveODE)
    solution_set = solveODE_v(samples.T)
    solution_set = jnp.reshape(solution_set, (x.shape[0], x.shape[0], solution_set.shape[1], 2))
    distance_mat = square_distance_tensor(solution_set)

    sample_grid = jnp.reshape(samples.T, (x.shape[0], y.shape[0], 2))

    for j in range(mcmc_samples):
        # Generate Data
        key, subkey = random.split(key)
        true_init, observation_times, observations = construct_observations(std, subkey, parameters)
        true_solution = SolveODE(true_init)

        # Construct Distribution Grids
        grid_ll = construct_log_likelihood_grid(observation_times, observations)
        grid_pt = grid_ll - jnp.log(weight_matrix)
        grid_ps = jnp.exp(grid_ll - jnp.max(grid_ll))
        grid_ps = grid_ps / jnp.sum(grid_ps)

        # Find maximum of grid for ML + MAP
        ml_est = max_estimate(grid_ll)
        map_state_est = ml_est
        map_trajectory_est = max_estimate(grid_pt)

        ml_trajectory = SolveODE(ml_est)
        map_trajectory = SolveODE(map_trajectory_est)

        # Compute MMSE initial condition
        mmse_initial_cond = sample_grid * grid_ps[:,:,None]
        mmse_initial_cond = jnp.sum(mmse_initial_cond, axis=(0,1))
        mmse_traject_init = SolveODE(mmse_initial_cond)

        # Construct MMSE trajectory
        mmse_trajectory_est = solution_set * grid_ps[:,:, None, None]
        mmse_trajectory_est = jnp.sum(mmse_trajectory_est, axis=(0,1))

        # Construct MMSE trajectory on manifold
        mse_mat = mse_with_distance(grid_ps, distance_mat)
        mmse_init = min_estimate(mse_mat)
        mmse_trajectory_proj = SolveODE(mmse_init)

        # Compute Error
        t_mse_MMSE[i] += jnp.sum(jnp.square(mmse_trajectory_est - true_solution))
        t_mae_MMSE[i] += jnp.sum(jnp.abs(mmse_trajectory_est - true_solution))
        t_sup_MMSE[i] += jnp.max(jnp.abs(mmse_trajectory_est - true_solution))

        t_mse_MMSE_proj[i] += jnp.sum(jnp.square(mmse_trajectory_proj - true_solution))
        t_mae_MMSE_proj[i] += jnp.sum(jnp.abs(mmse_trajectory_proj - true_solution))
        t_sup_MMSE_proj[i] += jnp.max(jnp.abs(mmse_trajectory_proj - true_solution))

        t_mse_MMSE_init[i] += jnp.sum(jnp.square(mmse_traject_init - true_solution))
        t_mae_MMSE_init[i] += jnp.sum(jnp.abs(mmse_traject_init - true_solution))
        t_sup_MMSE_init[i] += jnp.max(jnp.abs(mmse_traject_init - true_solution))

        t_mse_ML[i] += jnp.sum(jnp.square(ml_trajectory - true_solution))
        t_mae_ML[i] += jnp.sum(jnp.abs(ml_trajectory - true_solution))
        t_sup_ML[i] += jnp.max(jnp.abs(ml_trajectory - true_solution))

        t_mse_MAP[i] += jnp.sum(jnp.square(map_trajectory - true_solution))
        t_mae_MAP[i] += jnp.sum(jnp.abs(map_trajectory - true_solution))
        t_sup_MAP[i] += jnp.max(jnp.abs(map_trajectory - true_solution))

        # Update Progress Bar
        inner_bar.update(1)

    t_mse_MMSE[i] /= mcmc_samples / parameters.step_size 
    t_mae_MMSE[i] /= mcmc_samples / parameters.step_size 
    t_sup_MMSE[i] /= mcmc_samples 
    t_mse_MMSE_proj[i] /= mcmc_samples / parameters.step_size 
    t_mae_MMSE_proj[i] /= mcmc_samples / parameters.step_size 
    t_sup_MMSE_proj[i] /= mcmc_samples 
    t_mse_MMSE_init[i] /= mcmc_samples / parameters.step_size 
    t_mae_MMSE_init[i] /= mcmc_samples / parameters.step_size 
    t_sup_MMSE_init[i] /= mcmc_samples 
    t_mse_ML[i]   /= mcmc_samples / parameters.step_size 
    t_mae_ML[i]   /= mcmc_samples / parameters.step_size 
    t_sup_ML[i]   /= mcmc_samples 
    t_mse_MAP[i]  /= mcmc_samples / parameters.step_size 
    t_mae_MAP[i]  /= mcmc_samples / parameters.step_size 
    t_sup_MAP[i]  /= mcmc_samples 
    outer_bar.update(1)

    

    
# %%
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

fig, axs = plt.subplots(2,2, sharex=True)

fig.set_figwidth(5.5)
fig.set_figheight(4)
axs[1,0].plot(time_horizons, t_mae_MMSE, label="MMSE Trajectory, Ambient")
axs[1,0].plot(time_horizons, t_mae_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,0].plot(time_horizons, t_mae_MAP, label="MAP Trajectory")
axs[1,0].plot(time_horizons, t_mae_ML, label="ML/MAP Initial Condition")
axs[1,0].plot(time_horizons, t_mae_MMSE_init, label="MMSE Initial Condition")
#axs[1,0].set_xscale('log')
#axs[1,0].set_yscale('log')
axs[1,0].set_title("Mean Absolute Error")
axs[1,0].set_xlim(2,30)
axs[1,0].set_ylim(0,30)

axs[1,1].plot(time_horizons, t_mse_MMSE, label="MMSE Trajectory, Ambient")
axs[1,1].plot(time_horizons, t_mse_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[1,1].plot(time_horizons, t_mse_MAP, label="MAP Trajectory")
axs[1,1].plot(time_horizons, t_mse_ML, label="ML/MAP Initial Condition")
axs[1,1].plot(time_horizons, t_mse_MMSE_init, label="MMSE Initial Condition")
#axs[1,1].set_xscale('log')
#axs[1,1].set_yscale('log')
axs[1,1].set_title("Mean Squared Error")
axs[1,1].set_xlim(2,30)
axs[1,1].set_ylim(0,30)

axs[0,0].plot(time_horizons, t_sup_MMSE, label="MMSE Trajectory, Ambient")
axs[0,0].plot(time_horizons, t_sup_MMSE_proj, label="MMSE Trajectory, Manifold")
axs[0,0].plot(time_horizons, t_sup_MAP, label="MAP Trajectory")
axs[0,0].plot(time_horizons, t_sup_ML, label="ML/MAP Initial Condition")
axs[0,0].plot(time_horizons, t_sup_MMSE_init, label="MMSE Initial Condition")
#axs[0,0].set_xscale('log')
#axs[0,0].set_yscale('log')
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
fig.show()
# %%

fig.savefig("quantitative_horizon.pdf", format="pdf", bbox_inches="tight")
# %%

jnp.savez("quant_horizon_lotka.npz", time_horizons=time_horizons,
                                    t_mse_MMSE      = jnp.array(t_mse_MMSE),
                                    t_mae_MMSE      = jnp.array(t_mae_MMSE),
                                    t_sup_MMSE      = jnp.array(t_sup_MMSE),
                                    t_mse_MMSE_proj = jnp.array(t_mse_MMSE_proj),
                                    t_mae_MMSE_proj = jnp.array(t_mae_MMSE_proj),
                                    t_sup_MMSE_proj = jnp.array(t_sup_MMSE_proj),
                                    t_mse_MMSE_init = jnp.array(t_mse_MMSE_init),
                                    t_mae_MMSE_init = jnp.array(t_mae_MMSE_init),
                                    t_sup_MMSE_init = jnp.array(t_sup_MMSE_init),
                                    t_mse_ML        = jnp.array(t_mse_ML),
                                    t_mae_ML        = jnp.array(t_mae_ML),
                                    t_sup_ML        = jnp.array(t_sup_ML),
                                    t_mse_MAP       = jnp.array(t_mse_MAP),
                                    t_mae_MAP       = jnp.array(t_mae_MAP),
                                    t_sup_MAP       = jnp.array(t_sup_MAP))
# %%
