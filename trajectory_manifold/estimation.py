"""This module includes functions for statistical estimation"""

from typing import Callable
from jaxtyping import Float, Array
from jax import jit
from manifold import system_pushforward_weight

def ML_estimation_state(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
) -> Float[Array, " dim"]:
    """Computes the maximum likelihood estimate of the state of a system.
    
    Given a differential equation, a likelihood function for the observations,
    the set of observations, and the observation times, compute the maximum
    likelihood estimate of the state of the system at the final observation time.

    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.

    Returns:
        The maximum likelihood estimate of the final state of the system.
    """
    likelihood = trajectory_likelihood(vector_field, 
                                       observations,
                                       observation_times, 
                                       observation_likelihood)

    likelihood = jit(likelihood) # Maps initial conditions to
    # Maximize likelihood
    pass 


def trajectory_likelihood(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
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
    pass

def state_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
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


def trajectory_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
    time_horizon
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
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.

    Returns:
        A function mapping the state at the initial observation time to the 
        likelihood of the observation.
    """
    posterior = state_posterior(vector_field,
                                observations,
                                observation_times,
                                observation_likelihood,
                                state_prior)
    
    def weight(initial_condition):
        return system_pushforward_weight(vector_field,
                                         time_horizon,
                                         initial_condition)
      
    return posterior


def MAP_estimation_state(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
) -> Float[Array, " dim"]:
    """Computes the MAP estimate of the state of a system.
    
    Given a differential equation, a likelihood function for the observations,
    the set of observations, and the observation times, compute the maximum
    likelihood estimate of the state of the system at the initial observation time.

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
        The MAP estimate of the initial state of the system.
    """
    posterior = state_posterior(vector_field, 
                                observations,
                                observation_times, 
                                observation_likelihood,
                                state_prior)
              
    posterior = jit(posterior)

    # Optimize the posterior
    pass 


def MAP_estimation_manifold(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
    time_interval: tuple[float, float],
) -> Float[Array, " dim"]:
    """Computes the MAP estimate of the trajectory of a system.
    
    Given a differential equation, a likelihood function for the observations,
    the set of observations, and the observation times, compute the maximum
    likelihood estimate of the state of the system at the final observation time.

    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.
        time_interval: The interval of time on which to compute the estimate.

    Returns:
        The MAP estimate of the final state of the system on the trajectory 
        manifold.
    """
    pass 


def MMSE_estimation_state(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
) -> Float[Array, " dim"]:
    """Computes the MMSE estimate of the state of a system.
    
    Given a differential equation, a likelihood function for the observations,
    the set of observations, and the observation times, compute the maximum
    likelihood estimate of the state of the system at the final observation time.

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
        The MMSE estimate of the final state of the system on the trajectory 
        manifold.
    """
    pass


def MMSE_ambient_estimation(
    vector_field: Callable[[any, Float[Array, " dim"], any], Float[Array, " dim"]], 
    observations: Float[Array, " observation_count dim2"],
    observation_times: Float[Array, " observation_count"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    state_prior: Callable[[Float[Array, " dim"]], float],
    time_interval: tuple[float, float],
    timesteps: int,
) -> Float[Array, " timesteps dim"]:
    """Computes the MMSE estimate of the trajectory of a system.
    
    Given a differential equation, a likelihood function for the observations,
    the set of observations, and the observation times, compute the maximum
    likelihood estimate of the state of the system at the final observation time.

    Args:
        vector_field: Governing differential equation mapping the current state
          to the derivative.
        observations: An N by K dimensional array of observations.
        observation_times: An N dimensional array of observation times.
        observation_likelihood: A function mapping pairs of states and 
          observations to the likelihood.
        state_prior: A function representing the prior distribution of the
          state of the system at the initial observation time.
        time_interval: The interval of time on which to compute the estimate.
        timesteps: The number of timesteps in the solution interval.

    Returns:
        The MMSE estimate of the final state of the system on the trajectory 
        manifold.
    """
    pass


def karcher_mean_estimation():
    pass