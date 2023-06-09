""" Computing geometric transformations onto the trajectory manifold.

This module contains the core functions requires for geometric transformations
of quantities onto the trajectory manifold through the use of an ODE
solver and a known vector field.
"""

from jax import jit, jacrev
import jax.numpy as jnp

from jaxtyping import Float, Array, PyTree
from typing import Callable

from diffrax import AbstractSolver, Heun
from diffrax import ODETerm, SaveAt, PIDController, diffeqsolve
from diffrax import AbstractStepSizeController

from functools import partial
from typing import NamedTuple

from .helpers import trapezoidal_correlation, trapezoidal_correlation_weighted
from .helpers import safe_det


class SolverParameters(NamedTuple):
    """Stores Information for ODE Solvers.
    
    Records the parameters for solving an ODE using Diffrax,
    including the solver, tolerances, output grid size, and time horizon.
    
    Attributes:
        stepsize_controller: Absolute tolerance for the ODE solution.
        step_size_internal: Internal mesh size for constant step size controller.
        step_size_output: Output mesh size. Note: Does not impact internal computations.
        time_interval: Tuple of (initial time, final time). The interval is 
          inclusive of the initial time, but exclusive of the final time.
        solver: The particular ODE solver to use.
        max_steps: Max steps for the solver.
    """

    stepsize_controller: AbstractStepSizeController
    step_size_internal: Float
    step_size_output: Float
    time_interval: tuple[Float, Float]
    solver: AbstractSolver
    max_steps: int


@partial(jit, static_argnames=['vector_field', 'solver_parameters'])
def system_sensitivity_and_solution(
    vector_field: Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    initial_condition: Float[Array, " dim"],
    system_parameters: PyTree,
    solver_parameters: SolverParameters,
) -> tuple[tuple[Float[Array, " dim timesteps dim"], PyTree],
           Float[Array, " timesteps dim"]]:
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
        `(sensitivity, solution)`

        sensitivity: A matrix where each row represents the sensitivity of the 
          system solution to a perturbation along some element of an orthonormal 
          basis of the state space.
        solution: A matrix representing the corresponding solution of the ODE
    """

    term = ODETerm(vector_field)
    solver = solver_parameters.solver
    timesteps = jnp.arange(solver_parameters.time_interval[0], 
                           solver_parameters.time_interval[1], 
                           step=solver_parameters.step_size_output)

    saveat = SaveAt(ts = timesteps)
    stepsize_controller = solver_parameters.stepsize_controller

    @jit
    def diffeq_solution(
        x0: Float[Array, " dim"],
        p: PyTree
    ) -> Float[Array, " timesteps dim"]:
        """Returns the solution to the differential equation."""
        return diffeqsolve(term,
                           solver,
                           t0  = solver_parameters.time_interval[0],
                           t1  = solver_parameters.time_interval[1],
                           dt0 = solver_parameters.step_size_internal,
                           saveat = saveat,
                           stepsize_controller = stepsize_controller,
                           y0 = x0,
                           args = p,
                           max_steps = solver_parameters.max_steps).ys

    solution = diffeq_solution(initial_condition, system_parameters)
    sensitivity = jacrev(diffeq_solution, argnums=[0,1])(initial_condition, 
                                                         system_parameters)

    if isinstance(system_parameters, jnp.ndarray):
      sensitivity = (jnp.moveaxis(sensitivity[0], 2, 0), 
                     jnp.moveaxis(sensitivity[1], 2, 0))
    else:
      sensitivity = (jnp.moveaxis(sensitivity[0], 2, 0), 
                     sensitivity[1])

    return (sensitivity, solution)


@partial(jit, static_argnames=['vector_field', 'solver_parameters'])
def system_sensitivity(
    vector_field: Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    initial_condition: Float[Array, " dim"],
    system_parameters: PyTree,
    solver_parameters: SolverParameters,
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
        system_parameters: The parameters for the vector field.
        solver_parameters: The set of parameters for the ODE solver.
    
    Returns:
        A matrix where each row represents the sensitivity of the system solution
        to a perturbation along some element of an orthonormal basis of the
        state space.
    """

    term = ODETerm(vector_field)
    solver = solver_parameters.solver
    timesteps = jnp.arange(solver_parameters.time_interval[0], 
                           solver_parameters.time_interval[1], 
                           step=solver_parameters.step_size_output)

    saveat = SaveAt(ts = timesteps)
    stepsize_controller = solver_parameters.stepsize_controller

    @jit
    def diffeq_solution(
        x0: Float[Array, " dim"],
        p: PyTree
    ) -> Float[Array, " timesteps dim"]:
        """Returns the solution to the differential equation."""
        return diffeqsolve(term,
                           solver,
                           t0  = solver_parameters.time_interval[0],
                           t1  = solver_parameters.time_interval[1],
                           dt0 = solver_parameters.step_size_internal,
                           saveat = saveat,
                           stepsize_controller = stepsize_controller,
                           y0 = x0,
                           args = p,
                           max_steps=solver_parameters.max_steps).ys

    sensitivity = jacrev(diffeq_solution, argnums=[0,1])(initial_condition, 
                                                         system_parameters)

    if isinstance(system_parameters, jnp.ndarray):
      return (jnp.moveaxis(sensitivity[0], 2, 0), jnp.moveaxis(sensitivity[1], 2, 0))
    else:
      return (jnp.moveaxis(sensitivity[0], 2, 0), sensitivity[1])


@partial(jit, static_argnames=['vector_field', 'time_interval'])
def system_pushforward_weight(
    vector_field: Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    time_interval: tuple[float, float], 
    initial_condition: Float[Array, " dim"],
    system_parameters: Float[Array, " dim2"],
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
        initial_condition: The position in the statespace to be pushed onto 
          the manifold.
        system_parameters: The parameters for the vector field.
        
    Returns:
        The weight required to push a density onto the trajectory manifold.
    """

    absolute_tolerance = 1e-2
    relative_tolerance = 1e-2
    max_steps = 16**4
    step_size = 0.01
    solver = Heun()
    parameters = SolverParameters(stepsize_controller=PIDController(rtol=relative_tolerance,
                                                                    atol=absolute_tolerance),
                                  step_size_internal=0.1,
                                  step_size_output=step_size,
                                  time_interval = time_interval,
                                  solver=solver,
                                  max_steps = max_steps)

    U = system_sensitivity(vector_field, initial_condition, system_parameters, parameters)
    A = trapezoidal_correlation(U[0], step_size)
    B = trapezoidal_correlation(U[1], step_size)

    return jnp.sqrt(abs(safe_det(A) * safe_det(B)))


@partial(jit, static_argnames=['vector_field', 'time_interval'])
def system_pushforward_weight_reweighted(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    time_interval: Float, 
    initial_condition: Float[Array, " dim"],
    system_parameters: Float[Array, " dim2"],
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
        initial_condition: The position in the statespace to be pushed onto 
          the manifold.
        system_parameters: The parameters for the vector field.
        step_size: The step size for the numerical solution of the ODE.
        kernel: An N by K by K array of N timesteps of an integral kernel to
          apply to a K dimensional space.
        
    Returns:
        The weight required to push a density onto the trajectory manifold.
    """

    absolute_tolerance = 1e-2
    relative_tolerance = 1e-2
    max_steps = 16**4
    step_size = 0.01
    solver = Heun()
    parameters = SolverParameters(stepsize_controller=PIDController(rtol=relative_tolerance,
                                                                    atol=absolute_tolerance),
                                  step_size_internal=0.1,
                                  step_size_output=step_size,
                                  time_interval = time_interval,
                                  solver=solver,
                                  max_steps = max_steps)

    U = system_sensitivity(vector_field, initial_condition, system_parameters, parameters)
    A = trapezoidal_correlation_weighted(U, step_size, kernel)

    return jnp.sqrt(abs(jnp.linalg.det(A)))