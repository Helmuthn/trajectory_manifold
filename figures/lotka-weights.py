import trajectory_manifold.examples as examples
from trajectory_manifold.manifold import SolverParameters
from trajectory_manifold.manifold import system_pushforward_weight

from diffrax import  Heun, PIDController
import jax.numpy as jnp
from jax import jit, vmap
import numpy as onp

from tqdm import tqdm

## Setup Problem Specification

vector_field = examples.lotka_volterra_vector_field(1,2,4,2)
center = 1.2 # Center of sample grid
delta = 0.05 # Grid Step Size

stepsize_controller = PIDController(rtol = 1e-2,
                                    atol = 1e-2)

parameters = SolverParameters(stepsize_controller=stepsize_controller,
                              step_size_internal = 0.1,
                              step_size_output = 0.1,
                              time_interval = (0,10),
                              solver=Heun(),
                              max_steps=16**5)

## Compute Trajectory Weight Grid

x = y = jnp.arange(center - 1, center + 1 + delta, delta)
X, Y = jnp.meshgrid(x, y)
samples = jnp.stack([X.flatten(), Y.flatten()])

def lotka_weight(state):
    """Return an approximation of the pushforward weight for an initial condition."""
    return system_pushforward_weight(vector_field, parameters.time_interval, state, jnp.zeros(0))

lotka_weight = jit(lotka_weight)
vec_lotka_weight = vmap(lotka_weight, 1)
vec_lotka_weight = jit(vec_lotka_weight)

weight_matrix = onp.zeros(samples.shape[1])
chunk_size = 5
for i in tqdm(range(samples.shape[1]//chunk_size)):
    weight_matrix[i*chunk_size:(i+1)*chunk_size] = vec_lotka_weight(samples[:,i*chunk_size:(i+1)*chunk_size])
start = chunk_size * (samples.shape[1]//chunk_size)
weight_matrix[start:] = vec_lotka_weight(samples[:, start:])

# Save Weight Matrix
jnp.savez("lotkaweights.npz",   samples = samples, 
                                weight_matrix = weight_matrix,
                                x = x,
                                y = y)
