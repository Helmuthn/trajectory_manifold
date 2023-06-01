import trajectory_manifold.examples as examples
from trajectory_manifold.manifold import SolverParameters
from trajectory_manifold.manifold import system_pushforward_weight, system_sensitivity
from trajectory_manifold.helpers import trapezoidal_correlation

from diffrax import  Heun
import jax.numpy as jnp
from jax import jit, vmap
import numpy as onp

from tqdm import tqdm

## Setup Problem Specification

vector_field = examples.lotka_volterra_vector_field(1,2,4,2)
center = 1.2 # Center of sample grid
delta = 0.05 # Grid Step Size

parameters = SolverParameters(relative_tolerance = 1e-2,
                              absolute_tolerance = 1e-2,
                              step_size = 0.1,
                              time_interval = (0,30),
                              solver=Heun(),
                              max_steps=16**5)

## Compute Trajectory Weight Grid

x = y = jnp.arange(center - 1, center + 1 + delta, delta)
X, Y = jnp.meshgrid(x, y)
samples = jnp.stack([X.flatten(), Y.flatten()])

def lotka_sensitivity(state):
    """Return an approximation of the sensitivity for an initial condition."""
    return system_sensitivity(vector_field, state, parameters)

lotka_weight = jit(lotka_sensitivity)
vec_lotka_weight = vmap(lotka_weight, 1)
vec_lotka_weight = jit(vec_lotka_weight)

timesteps = jnp.arange(parameters.time_interval[0], parameters.time_interval[1], step=parameters.step_size_output)
weight_matrix = onp.zeros((samples.shape[1], 2, timesteps.shape[0], 2))
chunk_size = 2
for i in tqdm(range(samples.shape[1]//chunk_size)):
    weight_matrix[i*chunk_size:(i+1)*chunk_size, :, :] = vec_lotka_weight(samples[:,i*chunk_size:(i+1)*chunk_size])
start = chunk_size * (samples.shape[1]//chunk_size)
weight_matrix[start:, :, :, :] = vec_lotka_weight(samples[:, start:])

# Save Weight Matrix
time_horizons = onp.arange(2,32,2)
out = onp.zeros((samples.shape[1], time_horizons.shape[0]))

for i in tqdm(range(time_horizons.shape[0])):
    steps = time_horizons[i] * 10
    for j in tqdm(range(samples.shape[1])):
        sensitivity = weight_matrix[j,:,:steps,:] 
        A = trapezoidal_correlation(sensitivity, 0.1)
        out[j, i] = jnp.sqrt(abs(jnp.linalg.det(A)))

jnp.savez("lotka-weights-full.npz", samples = samples,
                                    weight_matrix = out,
                                    x = x,
                                    y = y,
                                    time_horizons = time_horizons)
