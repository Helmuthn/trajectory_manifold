#%%
import trajectory_manifold.estimation as estimation
import trajectory_manifold.examples as examples
from trajectory_manifold.manifold import SolverParameters
from trajectory_manifold.optimize import distance_gradient

from math import pi

import matplotlib.pyplot as plt
from diffrax import ODETerm, SaveAt, PIDController, diffeqsolve, Heun

import jax.numpy as jnp
from jax import random, jit, vmap

import numpy as onp

import optax
from tqdm import tqdm
import imageio.v3 as iio
from pygifsicle import optimize

#%%

## Setup Problem Specification

vector_field = examples.lotka_volterra_vector_field(1,2,4,2)
dimension = 2
noise_std = 1 # Noise Power
center = 1.2 # Center of sample grid
delta = 0.05 # Grid Step Size

parameters = SolverParameters(relative_tolerance = 1e-2,
                              absolute_tolerance = 1e-2,
                              step_size = 0.1,
                              time_interval = (0,10),
                              solver=Heun(),
                              max_steps=16**5)


## Simulate Observations
print("Simulating Observations")

key = random.PRNGKey(123)
key, subkey = random.split(key)
true_init = 2 * random.uniform(subkey, shape=(dimension,)) + center - 1

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
noise = noise_std*random.normal(subkey, shape=states.shape)
observations = states + noise
observation_times = observation_times[:30]
observations = observations[:30,:]
subsample = 6

## Compute Estimation Functions
print("Construction Estimates")

def observation_log_likelihood(observation, state):
    partition = jnp.power(2 * pi, -observations.shape[1]/2.0)
    return jnp.log(partition) -1 * jnp.sum(jnp.square(observation - state))/(2*noise_std**2)

def state_log_prior(state):
    return -1 * jnp.log(9)


log_posterior_state = estimation.state_log_posterior(vector_field,
                                                     observations[::subsample],
                                                     observation_times[::subsample], 
                                                     observation_log_likelihood,
                                                     state_log_prior,
                                                     parameters)

@jit
def posterior_state(state):
    return jnp.exp(log_posterior_state(state))

v_ps = vmap(posterior_state)

@jit
def SolveODE(initial_state):
    return diffeqsolve(term,
                     solver,
                     t0 = parameters.time_interval[0],
                     t1 = parameters.time_interval[1],
                     dt0 = 0.1,
                     saveat = saveat,
                     stepsize_controller = stepsize_controller,
                     y0 = initial_state).ys

solveODE_v = vmap(SolveODE)

SolveODE(true_init)

# Approximate expected value through importance sampling
#%%
sample_count = 100000
key, subkey = random.split(key)
importance_samples = 2 * random.uniform(subkey, shape=(sample_count, dimension,)) + center - 1
sample_weights = v_ps(importance_samples)
sample_sols = solveODE_v(importance_samples) 

estimate = jnp.sum(sample_weights[:,None,None] * sample_sols, axis=0)/jnp.sum(sample_weights)
truth = SolveODE(true_init)

# %%

plt.plot(estimate)
plt.plot(truth)
plt.show()
# %%
# Now, do a gradient descent to optimize
time = jnp.arange(parameters.time_interval[0], 
                               parameters.time_interval[1] + parameters.step_size, 
                               step=parameters.step_size)

start_learner_rate = 1e-1
optimizer = optax.adam(start_learner_rate)

params = jnp.ones(2) * .4
opt_state = optimizer.init(params)

g = lambda init: distance_gradient(init,
                                   vector_field,
                                   estimate,
                                   parameters)
g = jit(g)

step_count = 150
error = onp.zeros(step_count)
error_state = onp.zeros(step_count)
for i in tqdm(range(step_count)):
    grads = g(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)[0]

    # Plotting
    trajectory_est = SolveODE(params)
    error[i] = jnp.sum(jnp.square(truth - trajectory_est)) * parameters.step_size
    error_state[i] = jnp.sum(jnp.square(params - true_init))
    filename = f'data/{i}.png'
    plt.plot(time, truth, linestyle="--")
    plt.gca().set_prop_cycle(None)
    plt.plot(time, trajectory_est)
    plt.axvline(x=observation_times[::subsample][-1], linestyle="dotted", color="red")
    plt.xlim(0,10)
    plt.ylim(0,3.5)
    plt.xlabel("Time")
    plt.ylabel("Population")

    plt.savefig(filename)
    plt.close()


#%%
frames = onp.stack([iio.imread(f"data/{i}.png") for i in range(step_count)], axis=0)
iio.imwrite("optimize.gif", frames)
optimize("optimize.gif")

# %%

plt.plot(error, label="Trajectory")
plt.plot(error_state, label="Initial Condition")
plt.xlabel("Gradient Descent Step")
plt.ylabel("Squared Error")
plt.yscale("log")
plt.legend()
plt.show()
# %%
