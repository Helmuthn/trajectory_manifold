from dataclasses import dataclass 

from jax import jit
from jax.lax import fori_loop
from jax.numpy import dot, zeros, sqrt
from jax.numpy.linalg import det
import jax.numpy as jnp

from jaxtyping import Float, Array
from typing import Callable

from diffrax import AbstractSolver, Tsit5


@dataclass
class SolverParameters:
    """Stores Information for ODE Solvers.
    
    Records the parameters for solving an ODE using Diffrax,
    including the solver, tolerances, output grid size, and time horizon
    
    Attributes:
        relative_tolerance: Relative tolerance for the ODE solution
        absolute_tolerance: Absolute tolerance for the ODE solution
        step_size: Output mesh size. Note: Does not impact internal computations.
        time_horizon: Length of the solution in seconds.
        solver: The particular ODE solver to use.
    """

    relative_tolerance: float
    absolute_tolerance: float
    step_size: float
    time_horizon: float
    solver: AbstractSolver

    
@jit
def frobenius_inner_product(
    x: Float[Array, " *dim"], 
    y: Float[Array, " *dim"], 
) -> Float:
    """Computes the Frobenius inner product of two matrices.

    Given two multidimensional arrays, computes the sum of
    the elementwise product of the arrays.
    
    Args:
        x: A multidimensional array.
        y: A multidimensional array.
    
    Returns:
        The sum of the elementwise product of x and y.
    """
    return jnp.sum(x * y)


@jit
def trapezoidal_inner_product(
    x: Float[Array, " timesteps dim"], 
    y: Float[Array, " timesteps dim"], 
    step_size: Float,
) -> Float:
    """Approximate the inner product by the trapezoidal rule.
    
    Computes an approximate inner product between two functions represented
    by a finite-grid approximation with a fixed step size.
    Computed through the trapezoidal integration scheme.
    
    Args:
        x: Grid approximation of the first function. Each row represents the 
          value of the multivariate function at a given timestep.
        y: Grid approximation of the second function. Each row represents the
          value of the multivariate function at a given timestep.
        step_size: Spacing between sample points in the functions.
        
    Returns:
        An approximation of the L2 inner product.
    """

    out = (dot(x[0, :],  y[0, :]) + dot(x[-1, :], y[-1, :])) / 2
    out += frobenius_inner_product(x[1:-1, :], y[1:-1, :])
    return out * step_size


@jit
def trapezoidal_correlation(
    U: Float[Array, " functions timesteps dim"],
    step_size: Float,
) -> Float[Array, " functions functions"]:
    """Computes the inner products between rows of the given matrix.
    
    Constructs an M by M matrix of approximate inner products between M 
    multi-variate functions computed using N evenly spaced samples in a 
    trapezoidal integration scheme. 
    
    Args:
        U: M by N by K matrix representing N samples each of M functions that
          take values in a K-dimensional space.
        step_size: Spacing between sample points in the functions.
    
    Returns:
        An M by M matrix where the (i,j)'th element is the approximate
        inner product between rows i and j of the input matrix. 
    """

    M = U.shape[0]
    out = zeros((M, M))

    def inner(i, val):
        out, U, step_size = val

        def inner2(j, val):
            out, U, step_size, i = val
            value = trapezoidal_inner_product(U[i,...], U[j,...], step_size)
            out = out.at[i,j].set(value)
            out = out.at[j,i].set(value)
            return (out, U, step_size, i)

        out, U, step_size, i = fori_loop(i, M, inner2, (out, U, step_size, i))

        return out, U, step_size

    out, U, step_size = fori_loop(0, M, inner, (out, U, step_size))

    return out


def system_sensitivity(
    vector_field: Callable[[Float[Array, " dim"]], Float[Array, " dim"]], 
    initial_condition: Float[Array, " dim"],
    parameters: SolverParameters,
    ) -> Float[Array, " dim timesteps"]:
    """Computes the differential equation sensitivity to the initial conditions.
    
    Given a differential equation, initial condition, and desired time horizon,
    computes the Jacobian of the transformation from the initial condition
    to the Riemannian manifold of valid trajectories. The Jacobian is expressed
    in the ambient space of square integrable functions.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        initial_condition: The position in the statespace to be pushed onto 
          the manifold.
        parameters: The set of parameters for the ODE solver.
    
    Returns:
        A matrix where each row represents the sensitivity of the system solution
        to a perturbation along some element of an orthonormal basis of the
        state space.
    """
    pass


def system_pushforward_weight(
    vector_field: Callable[[Float[Array, " dim"]], Float[Array, " dim"]], 
    time_horizon: Float, 
    initial_condition: Float[Array, " dim"]
    ) -> Float:
    """Computes the pushforward weight for a given initial condition.
    
    Given a differential equation, initial condition, and desired time horizon,
    computes the weight required to push densities onto the Riemannian manifold
    of valid trajectories of the system.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        time_horizon: Time horizon for the trajectory manifold.
        initial_conditon: The position in the statespace to be pushed onto 
          the manifold.
        
    Returns:
        The weight required to push a density onto the trajectory manifold.
    """

    absolute_tolerance = 1e-4
    relative_tolerance = 1e-4
    step_size = 0.01
    solver = Tsit5()
    parameters = SolverParameters(relative_tolerance, 
                                  absolute_tolerance, 
                                  step_size, 
                                  time_horizon, 
                                  solver)

    U = system_sensitivity(vector_field, initial_condition, parameters)
    A = trapezoidal_correlation(U)

    return sqrt(abs(det(A)))