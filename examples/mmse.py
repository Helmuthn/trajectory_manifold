import matplotlib.pyplot as plt

from trajectory_manifold import examples
from trajectory_manifold.manifold import SolverParameters
from diffrax import Heun, ConstantStepSize

vector_field = examples.lotka_volterra_vector_field(1,2,4,2)

parameters = SolverParameters(ConstantStepSize(),
                              step_size = 0.1,
                              time_interval = (0,10.1),
                              solver=Heun(),
                              max_steps=16**5)

# Diffrax Setup

from diffrax import ODETerm, SaveAt, PIDController, diffeqsolve
from jax import jit, vmap
import jax.numpy as jnp

term = ODETerm(vector_field)
solver = parameters.solver
observation_times = jnp.arange(parameters.time_interval[0],
                               parameters.time_interval[1],
                               step=parameters.step_size_output)

saveat = SaveAt(ts = observation_times)

stepsize_controller = parameters.stepsize_controller

@jit
def SolveODE(initial_state):
    return diffeqsolve(term,
                       solver,
                       t0 = parameters.time_interval[0],
                       t1 = parameters.time_interval[1],
                       dt0 = parameters.step_size_internal,
                       saveat = saveat,
                       stepsize_controller = stepsize_controller,
                       y0 = initial_state).ys

solveODE_v = vmap(SolveODE)



# Observation Likelihoods
from math import pi

def observation_log_likelihood(observation, state):
    """Compute log p(y|x) for a given observation and state"""
    partition = jnp.power(2 * pi, -observations.shape[1]/2.0)
    return jnp.log(partition) - jnp.sum(jnp.square(observation - state))/2

def state_log_prior(state):
    """Compute log p(x) for a given state"""
    return -1 * jnp.log(9)

# Generate Observations
from jax import random

dimension = 2
subsample = 6
center = 1.2

key = random.PRNGKey(123)
key, subkey = random.split(key)
true_init = 2 * random.uniform(subkey, shape=(dimension,)) + center - 1

states = SolveODE(true_init)

key, subkey = random.split(key)
noise = random.normal(subkey, shape=states.shape)

observations = states + noise
observation_times = observation_times[:30:subsample]
observations = observations[:30:subsample,:]


step_times = jnp.arange(parameters.time_interval[0],
                        parameters.time_interval[1],
                        step=parameters.step_size_output)

fig, ax = plt.subplots()
ax.plot(step_times, states[:,0])
ax.plot(step_times, states[:,1])
ax.scatter(observation_times, observations[:,0])
ax.scatter(observation_times, observations[:,1])
ax.set_ylim(jnp.min(observations)-.1, jnp.max(observations)+0.2)
ax.set_xlim(0, 10)
ax.vlines(observation_times[-1], jnp.min(observations)-.1, jnp.max(observations)+.2, color="red", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
fig.set_figwidth(6)
fig.set_figheight(4)
fig.savefig("observations.svg")


# Find Posterior
from trajectory_manifold import estimation

log_posterior_state = estimation.state_log_posterior(vector_field,
                                                     observations,
                                                     observation_times,
                                                     observation_log_likelihood,
                                                     state_log_prior,
                                                     parameters)

@jit
def posterior_state(state):
    return jnp.exp(log_posterior_state(state))

posterior_state_v = vmap(posterior_state)

# Importance Sampling
sample_count = 100000
key, subkey = random.split(key)
samples = 2 * random.uniform(subkey, shape=(sample_count, dimension,)) + center - 1

sample_weights = posterior_state_v(samples)
sample_sols = solveODE_v(samples)

estimate = jnp.sum(sample_weights[:,None,None] * sample_sols, axis=0)/jnp.sum(sample_weights)

fig, ax = plt.subplots()
ax.plot(step_times, states[:,0], linestyle="--")
ax.plot(step_times, states[:,1], linestyle="--")
ax.set_prop_cycle(None)
ax.plot(step_times, estimate[:,0])
ax.plot(step_times, estimate[:,1])
ax.set_ylim(0, jnp.max(observations)+0.2)
ax.set_xlim(0, 10)
ax.vlines(observation_times[-1], 0, jnp.max(observations)+.2, color="red", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
fig.set_figwidth(6)
fig.set_figheight(4)
fig.savefig("mmse.svg")

# Project
from trajectory_manifold import optimize

g = lambda state: optimize.distance_gradient(state,
                                             vector_field,
                                             estimate,
                                             parameters)
g = jit(g)

import optax

start_learner_rate = 1e-1
optimizer = optax.adam(start_learner_rate)

state = jnp.ones(2) * .4
opt_state = optimizer.init(state)

step_count = 100

for i in range(step_count):
    grads = g(state)
    updates, opt_state = optimizer.update(grads, opt_state)
    state = optax.apply_updates(state, updates)[0]

trajectory_estimate = SolveODE(state)


fig, ax = plt.subplots()
ax.plot(step_times, states[:,0], linestyle="--")
ax.plot(step_times, states[:,1], linestyle="--")
ax.set_prop_cycle(None)
ax.plot(step_times, estimate[:,0], linestyle="dotted")
ax.plot(step_times, estimate[:,1], linestyle="dotted")
ax.set_prop_cycle(None)
ax.plot(step_times, trajectory_estimate[:,0])
ax.plot(step_times, trajectory_estimate[:,1])
ax.set_ylim(0, jnp.max(observations)+0.2)
ax.set_xlim(0, 10)
ax.vlines(observation_times[-1], 0, jnp.max(observations)+.2, color="red", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
fig.set_figwidth(6)
fig.set_figheight(4)
fig.savefig("final.pdf")
