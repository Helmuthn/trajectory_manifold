"""This module is for MCMC optimization methods.

The main approach in this module is to use simulated annealing.
This is due to the dependence of the first-order derivative of the
posterior distribution of trajectories on the second-order sensitivity
of the solution to the initial conditions.
"""
from typing import Callable
from jaxtyping import Float, Array
import jax.numpy as jnp
from jax import random


def simulated_annealing(
    f: Callable,
    proposal_variable: Callable, # Random variable
    sample_count: int,
):
    """Optimize a given function through simulated annealing."""
    pass 


def zero_order_gradient_estimate(
        f: Callable,
        x: Float[Array, " dim"],
        smoothing: float,
        batch_size: int,
        key,
        
) -> Float[Array, " dim"]:
    """Construct a zero-order estimate of the gradient of a function."""
    center = f(x)
    samples = random.normal(key, x.shape + (batch_size,))
    magnitudes = f(x[:, None] + smoothing * samples) - center
    estimates = magnitudes * samples 
    return jnp.sum(estimates, axis=1) / smoothing / batch_size


def optimize_gradient(
    f: Callable,
):
    pass