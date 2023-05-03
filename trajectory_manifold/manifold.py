""" Computing geometric transformations onto the trajectory manifold.

This module contains the core functions requires for geometric transformations
of quantities onto the trajectory manifold through the use of an ODE
solver and a known vector field.
"""

from jax import jit, jacrev
import jax.numpy as jnp

from jaxtyping import Float, Array, PyTree
from typing import Callable

from diffrax import AbstractSolver, Tsit5, Heun
from diffrax import ODETerm, SaveAt, PIDController, diffeqsolve

from functools import partial
from typing import NamedTuple

from .helpers import trapezoidal_correlation, trapezoidal_correlation_weighted


class SolverParameters(NamedTuple):
    """Stores Information for ODE Solvers.
    
    Records the parameters for solving an ODE using Diffrax,
    including the solver, tolerances, output grid size, and time horizon
    
    Attributes:
        relative_tolerance: Relative tolerance for the ODE solution
        absolute_tolerance: Absolute tolerance for the ODE solution
        step_size: Output mesh size. Note: Does not impact internal computations.
        time_interval: tuple of (initial time, final time)
        solver: The particular ODE solver to use.
        max_steps: max steps for the solver
    """

    relative_tolerance: float
    absolute_tolerance: float
    step_size: float
    time_interval: tuple[float, float]
    solver: AbstractSolver
    max_steps: int


@partial(jit, static_argnames=['vector_field', 'parameters'])
def system_sensitivity(
    vector_field: Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    initial_condition: Float[Array, " dim"],
    parameters: SolverParameters,
) -> Float[Array, " dim timesteps dim"]:
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

    term = ODETerm(vector_field)
    solver = parameters.solver
    timesteps = jnp.arange(parameters.time_interval[0], 
                           parameters.time_interval[1] + parameters.step_size, 
                           step=parameters.step_size)

    saveat = SaveAt(ts = timesteps)
    stepsize_controller = PIDController(rtol = parameters.relative_tolerance,
                                        atol = parameters.absolute_tolerance)

    @jit
    def diffeq_solution(
        x0: Float[Array, " dim"],
    ) -> Float[Array, " timesteps dim"]:
        """Returns the solution to the differential equation."""
        return diffeqsolve(term,
                           solver,
                           t0 = parameters.time_interval[0],
                           t1 = parameters.time_interval[1],
                           dt0 = 0.1,
                           saveat = saveat,
                           stepsize_controller = stepsize_controller,
                           y0 = x0,
                           max_steps=parameters.max_steps).ys

    sensitivity = jacrev(diffeq_solution)(initial_condition)

    return jnp.moveaxis(sensitivity, 2, 0)

@partial(jit, static_argnames=['vector_field', 'time_interval'])
def system_pushforward_weight(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    time_interval: tuple[float, float], 
    initial_condition: Float[Array, " dim"],
) -> Float:
    """Computes the pushforward weight for a given initial condition.
    
    Given a differential equation, initial condition, and desired time horizon,
    computes the weight required to push densities onto the Riemannian manifold
    of valid trajectories of the system.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        time_interval: Time interval for the trajectory manifold in the form
          (initial time, final time).
        initial_conditon: The position in the statespace to be pushed onto 
          the manifold.
        
    Returns:
        The weight required to push a density onto the trajectory manifold.
    """

    absolute_tolerance = 1e-2
    relative_tolerance = 1e-2
    max_steps = 16**4
    step_size = 0.01
    solver = Heun()
    parameters = SolverParameters(relative_tolerance, 
                                  absolute_tolerance, 
                                  step_size, 
                                  time_interval, 
                                  solver,
                                  max_steps)

    U = system_sensitivity(vector_field, initial_condition, parameters)
    A = trapezoidal_correlation(U, step_size)

    return jnp.sqrt(abs(jnp.linalg.det(A)))


@partial(jit, static_argnames=['vector_field', 'time_interval'])
def system_pushforward_weight_reweighted(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    time_interval: Float, 
    initial_condition: Float[Array, " dim"],
    step_size: Float,
    kernel: Float[Array, " timesteps dim dim"]
) -> Float:
    """Computes the pushforward weight for a given initial condition.
    
    Given a differential equation, initial condition, and desired time horizon,
    computes the weight required to push densities onto the Riemannian manifold
    of valid trajectories of the system.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        time_interval: Time interval for the trajectory manifold.
        initial_conditon: The position in the statespace to be pushed onto 
          the manifold.
        step_size: The step size for the numerical solution of the ODE.
        kernel: An N by K by K array of N timesteps of an integral kernel to
          apply to a K dimensional space.
        
    Returns:
        The weight required to push a density onto the trajectory manifold.
    """

    absolute_tolerance = 1e-4
    relative_tolerance = 1e-4
    solver = Tsit5()
    parameters = SolverParameters(relative_tolerance, 
                                  absolute_tolerance, 
                                  step_size, 
                                  time_interval, 
                                  solver,
                                  16**4)

    U = system_sensitivity(vector_field, initial_condition, parameters)
    A = trapezoidal_correlation_weighted(U, step_size, kernel)

    return jnp.sqrt(abs(jnp.linalg.det(A)))