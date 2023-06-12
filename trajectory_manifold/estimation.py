"""This module includes functions for statistical estimation"""

from typing import Callable
from jaxtyping import Float, Array, PyTree
import jax
from .manifold import system_pushforward_weight, SolverParameters
from diffrax import ODETerm, SaveAt, diffeqsolve
import jax.numpy as jnp

def trajectory_log_likelihood(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_log_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"], PyTree], Float]:
    """Constructs a likelihood function for a set of observations of a system.
    
    Given a representation of a differential equation, a set of observation 
    times and an observation likelihood function dependent on the state, 
    construct a likelihood function jointly over all observations.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        parameters: Parameters for the ODE solver

    Returns:
        A function mapping the state and parameters at the initial observation 
        time to the log likelihood of the observation.
    """

    term = ODETerm(vector_field)
    solver = parameters.solver
    saveat = SaveAt(ts = observation_times)
    stepsize_controller = parameters.stepsize_controller
    t0 = parameters.time_interval[0]
    t1 = parameters.time_interval[1]
    dt0 = parameters.step_size_internal
    max_steps = parameters.max_steps

    def likelihood(state: Float[Array, " dim"],
                   system_parameters: Float[Array, " dim3"]
        ) -> Float:

        states = diffeqsolve(term,
                             solver,
                             args = system_parameters,
                             t0 = t0,
                             t1 = t1,
                             dt0 = dt0,
                             saveat = saveat,
                             stepsize_controller = stepsize_controller,
                             y0 = state,
                             max_steps=max_steps).ys

        likelihood_v = jax.vmap(observation_log_likelihood)

        return jnp.sum(likelihood_v(observations, states))
    
    return likelihood

def trajectory_likelihood(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"], PyTree], Float]:
    """Constructs a likelihood function for a set of observations of a system.
    
    Given a representation of a differential equation, a set of observation 
    times and an observation likelihood function dependent on the state, 
    construct a likelihood function jointly over all observations.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        parameters: Parameters for the ODE solver

    Returns:
        A function mapping the state and parameters at the initial observation 
        time to the likelihood of the observation.
    """

    def likelihood(state: Float[Array, " dim"],
                   system_parameters: Float[Array, " dim3"]
        ) -> Float:
        term = ODETerm(vector_field)
        solver = parameters.solver
        saveat = SaveAt(ts = observation_times)
        stepsize_controller = parameters.stepsize_controller
        
        states = diffeqsolve(term,
                             solver,
                             args = system_parameters,
                             t0 = parameters.time_interval[0],
                             t1 = parameters.time_interval[1],
                             dt0 = parameters.step_size_internal,
                             saveat = saveat,
                             stepsize_controller = stepsize_controller,
                             y0 = state,
                             max_steps=parameters.max_steps).ys

        likelihood_v = jax.vmap(observation_likelihood)

        return jnp.prod(likelihood_v(observations, states))
    
    return likelihood


def state_log_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_log_prior: Callable[[Float[Array, " dim"]], float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"], PyTree], float]:
    """Constructs a posterior distribution for the initial state of the system.
    
    Given a representation of a differential equation, a set of observation 
    times and an observation likelihood function dependent on the state, 
    construct a posterior distribution jointly over all observations.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.
        parameters: Parameters for the ODE solver

    Returns:
        A function mapping the state and parameters at the initial observation 
        time to the log posterior of the trajectory.
    """
    log_likelihood = trajectory_log_likelihood(vector_field, 
                                       observations,
                                       observation_times, 
                                       observation_likelihood,
                                       parameters)

    def log_posterior(
        state: Float[Array, " dim"],
        system_parameters: Float[Array, " dim3"]
    ) -> Float:
        return log_likelihood(state, system_parameters) + state_log_prior(state)
      
    return log_posterior

def state_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
    parameters: SolverParameters
) -> Callable[[Float[Array, " dim"], PyTree], Float]:
    """Constructs a posterior distribution for the initial state of the system.
    
    Given a representation of a differential equation, a set of observation 
    times and an observation likelihood function dependent on the state, 
    construct a posterior distribution jointly over all observations.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.
        parameters: Parameters for the ODE solver

    Returns:
        A function mapping the state and initial condition at the initial 
        observation time to the posterior of the initial condition and parameters.
    """
    likelihood = trajectory_likelihood(vector_field, 
                                       observations,
                                       observation_times, 
                                       observation_likelihood,
                                       parameters)

    def posterior(
        state: Float[Array, " dim"],
        system_parameters: Float[Array, " dim3"]
    ) -> Float:
        return likelihood(state, system_parameters) * state_prior(state)
      
    return posterior


def trajectory_log_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_log_prior: Callable[[Float[Array, " dim"]], float],
    time_interval: tuple[float, float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"], PyTree], Float]:
    """Constructs a posterior distribution for system trajectories.
    
    Given a representation of a differential equation, a set of observation 
    times and an observation likelihood function dependent on the state, 
    construct a posterior distribution jointly over all observations.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.
        time_interval: Time interval for the trajectory manifold in the form
          (initial time, final time).
        parameters: Parameters for the ODE solver

    Returns:
        A function mapping the state and system parameters at the initial 
        observation time to the posterior distribution of the trajectory.
    """
    log_posterior = state_log_posterior(vector_field,
                                observations,
                                observation_times,
                                observation_likelihood,
                                state_log_prior,
                                parameters)
    
    def weight(initial_condition, system_parameters):
        return system_pushforward_weight(vector_field,
                                         time_interval,
                                         initial_condition,
                                         system_parameters)
    
      
    return lambda state, params: log_posterior(state, params) \
                                - jnp.log(weight(state, params))

def trajectory_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
    time_interval: tuple[float, float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"], PyTree], Float]:
    """Constructs a posterior distribution for system trajectories.
    
    Given a representation of a differential equation, a set of observation 
    times and an observation likelihood function dependent on the state, 
    construct a posterior distribution jointly over all observations.
    
    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.
        time_interval: Time interval for the trajectory manifold in the form
          (initial time, final time).
        parameters: Parameters for the ODE solver

    Returns:
        A function mapping the state at the initial observation time to the 
        posterior distribution of the observation.
    """
    posterior = state_posterior(vector_field,
                                observations,
                                observation_times,
                                observation_likelihood,
                                state_prior,
                                parameters)
    
    def weight(initial_condition, system_parameters):
        return system_pushforward_weight(vector_field,
                                         time_interval,
                                         initial_condition,
                                         system_parameters)
    
      
    return lambda state, params: posterior(state, params) / weight(state, params)
