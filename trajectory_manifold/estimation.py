"""This module includes functions for statistical estimation"""

from typing import Callable
from jaxtyping import Float, Array
import jax
from .manifold import system_pushforward_weight, SolverParameters
from diffrax import ODETerm, SaveAt, PIDController, diffeqsolve
import jax.numpy as jnp

def trajectory_log_likelihood(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_log_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"]], float]:
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

    Returns:
        A function mapping the state at the initial observation time to the 
        likelihood of the observation.
    """

    def likelihood(state: Float[Array, " dim"]) -> float:
        term = ODETerm(vector_field)
        solver = parameters.solver
        saveat = SaveAt(ts = observation_times)
        stepsize_controller = parameters.stepsize_controller
        
        states = diffeqsolve(term,
                             solver,
                             t0 = parameters.time_interval[0],
                             t1 = parameters.time_interval[1],
                             dt0 = parameters.step_size_internal,
                             saveat = saveat,
                             stepsize_controller = stepsize_controller,
                             y0 = state,
                             max_steps=parameters.max_steps).ys

        likelihood_v = jax.vmap(observation_log_likelihood)

        return jnp.sum(likelihood_v(observations, states))
    
    return likelihood

def trajectory_likelihood(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"]], float]:
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

    Returns:
        A function mapping the state at the initial observation time to the 
        likelihood of the observation.
    """

    def likelihood(state: Float[Array, " dim"]) -> float:
        term = ODETerm(vector_field)
        solver = parameters.solver
        saveat = SaveAt(ts = observation_times)
        stepsize_controller = parameters.stepsize_controller
        
        states = diffeqsolve(term,
                             solver,
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
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_log_prior: Callable[[Float[Array, " dim"]], float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"]], float]:
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

    Returns:
        A function mapping the state at the initial observation time to the 
        likelihood of the observation.
    """
    log_likelihood = trajectory_log_likelihood(vector_field, 
                                       observations,
                                       observation_times, 
                                       observation_likelihood,
                                       parameters)

    def log_posterior(
        state: Float[Array, " dim"]
    ) -> float:
        return log_likelihood(state) + state_log_prior(state)
      
    return log_posterior

def state_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
) -> Callable[[Float[Array, " dim"]], float]:
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

    Returns:
        A function mapping the state at the initial observation time to the 
        likelihood of the observation.
    """
    likelihood = trajectory_likelihood(vector_field, 
                                       observations,
                                       observation_times, 
                                       observation_likelihood)

    def posterior(
        state: Float[Array, " dim"]
    ) -> float:
        return likelihood(state) * state_prior(state)
      
    return posterior


def trajectory_log_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_log_prior: Callable[[Float[Array, " dim"]], float],
    time_interval: tuple[float, float],
    parameters: SolverParameters,
) -> Callable[[Float[Array, " dim"]], float]:
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

    Returns:
        A function mapping the state at the initial observation time to the 
        posterior distribution of the observation.
    """
    log_posterior = state_log_posterior(vector_field,
                                observations,
                                observation_times,
                                observation_likelihood,
                                state_log_prior,
                                parameters)
    
    def weight(initial_condition):
        return system_pushforward_weight(vector_field,
                                         time_interval,
                                         initial_condition)
    
      
    return lambda state: log_posterior(state) - jnp.log(weight(state))

def trajectory_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
    time_interval: tuple[float, float],
) -> Callable[[Float[Array, " dim"]], float]:
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

    Returns:
        A function mapping the state at the initial observation time to the 
        posterior distribution of the observation.
    """
    posterior = state_posterior(vector_field,
                                observations,
                                observation_times,
                                observation_likelihood,
                                state_prior)
    
    def weight(initial_condition):
        return system_pushforward_weight(vector_field,
                                         time_interval,
                                         initial_condition)
    
      
    return lambda state: posterior(state) * weight(state)
