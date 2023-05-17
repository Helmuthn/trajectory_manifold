"""This module contains functions useful for optimization.

This module contains useful functions for optimization on the
trajectory manifold. It enables the computation of the pullback
of gradients into the ambient space, as well as helper functions
for grid-based methods.
"""
from typing import Callable
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp
from jax import random
from .manifold import system_sensitivity_and_solution, SolverParameters
from .helpers import trapezoidal_matrix_product



def distance_gradient(
    initial_condition: Float[Array, " dim"],
    vector_field: Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]],
    trajectory: Float[Array, " dim dim2"],
    params: SolverParameters,
) -> Float[Array, " dim"]:
    """Computes the gradient of the squared distance to a chosen trajectory.
    
    Computes the pullback along the transformation from initial conditions
    to system trajectories of the gradient of the squared distance from
    a given sampled trajectory in the ambient space.

    The gradient in the trajectory manifold is the projection of the gradient
    in the ambient space onto the tangent space of the manifold.

    Args:
        initial_condition: The initial condition around which to linearize.
        vector_field: Vector field defining the differential equation.
        trajectory: A sampled function from which the distance is computed.
        params: Parameters for the ODE solvers.
    
    Returns:
        The gradient of the distance to a function"""
    sensitivity, solution = system_sensitivity_and_solution(vector_field,
                                                            initial_condition,
                                                            params)
    gradient_ambient = solution - trajectory
    gradient_statespace = trapezoidal_matrix_product(gradient_ambient[None, :, :], 
                                                     sensitivity,
                                                     params.step_size)
    return gradient_statespace
    

def zero_order_gradient_estimate(
    f: Callable[[Float[Array, " dim batch_size"]], Float[Array, " batch_size"]],
    x: Float[Array, " dim"],
    smoothing: float,
    batch_size: int,
    key: random.KeyArray,
) -> Float[Array, " dim"]:
    """Construct a zero-order estimate of the gradient of a function.
    
    Constructs an approximation of the gradient of a given function through
    a Monte Carlo method. Generates `batch_size` samples of Gaussian perturbations
    from the chosen point, and averages the estimates constructed by finite-difference
    approximations.
    Includes a smoothing factor which adjusts the variance of the Gaussian 
    samples.

    Args:
        f: A function of which the gradient is to be computed.
        x: The point at which the gradient is computed
        smoothing: Smoothing factor in the gradient computation, determines the
          ball size for the random samples
        batch_size: The number of random points used in the gradient estimate.
        key: PRNG key for Jax RNG functioanlity
    
    Returns:
        An approximation of the gradient of f at x

    Notes:
        For more information, see 

        S. Liu, P. -Y. Chen, B. Kailkhura, G. Zhang, A. O. Hero III and 
        P. K. Varshney, "A Primer on Zeroth-Order Optimization in Signal 
        Processing and Machine Learning: Principals, Recent Advances, 
        and Applications," in IEEE Signal Processing Magazine, vol. 37, 
        no. 5, pp. 43-54, Sept. 2020, doi: 10.1109/MSP.2020.3003837.
    """
    center = f(x)
    samples = random.normal(key, x.shape + (batch_size,))
    magnitudes = f(x[:, None] + smoothing * samples) - center
    estimates = magnitudes * samples 
    return jnp.sum(estimates, axis=1) / smoothing / batch_size


def grid_optimum(
        grid: Float[Array, " dim1 dim2"], 
        x: Float[Array, " dim1"], 
        y: Float[Array, " dim2"], 
        upper: bool = True,
    ) -> Float[Array, "2"]:
    """ Compute the optimal value in a grid.

    Given a matrix of values representing some objective function on a
    grid, compute the point with the maximum or minimum value.

    Args:
        grid: Sampled function evaluations
        x: Values in the first index
        y: Values for the second index
        upper: Maximize if `True`, Minimize if `False`

    Returns:
        An array containing the x, y coordininate of the optimal grid point.
    """
    if upper:
        peak = jnp.argmax(grid)
    else:
        peak = jnp.argmin(grid)

    peak_x = x[peak % x.shape[0]]
    peak_y = y[peak // y.shape[0]]
    return jnp.stack([peak_x, peak_y])
