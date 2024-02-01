"""This module includes functions for statistical estimation"""

from typing import Callable
from jaxtyping import Float, Array, PyTree, Int
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
    log_prior: Callable[[Float[Array, " dim"], PyTree], float],
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
        log_prior: A function representing the prior distribution of the
          initial state and parameters of the system.
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
        return log_likelihood(state, system_parameters) \
               + log_prior(state, parameters)
      
    return log_posterior

def state_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    prior: Callable[[Float[Array, " dim"], PyTree], float],
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
        prior: A function representing the prior distribution of the
          initial state and parameters of the system.
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
        return likelihood(state, system_parameters) \
                * prior(state, system_parameters)
      
    return posterior


def trajectory_log_posterior(
    vector_field: Callable[[any, Float[Array, " dim"], PyTree], Float[Array, " dim"]], 
    observations: Float[Array, " timesteps dim2"],
    observation_times: Float[Array, " timesteps"],
    observation_likelihood: Callable[[Float[Array, " dim2"], Float[Array, " dim"]], float],
    log_prior: Callable[[Float[Array, " dim"], PyTree], float],
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
        prior: A function representing the prior distribution of the
          initial state and parameters of the system.
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
                                log_prior,
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
    prior: Callable[[Float[Array, " dim"], PyTree], float],
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
        prior: A function representing the prior distribution of the
          initial state and parameters of the system.
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
                                prior,
                                parameters)
    
    def weight(initial_condition, system_parameters):
        return system_pushforward_weight(vector_field,
                                         time_interval,
                                         initial_condition,
                                         system_parameters)
    
      
    return lambda state, params: posterior(state, params) / weight(state, params)


def credible_trajectories(state_and_param, 
                          vector_field, 
                          posterior, 
                          parameters
        ) -> Float[Array, " dim"]:
    """Computes the set of credible trajectories
    
    Args:
      state_and_param: 
      vector_field:
      posterior:
      parameters:
      
    Returns:
      The list of credible trajectories in the set"""
    pass
    

def quantized_credible_regions(points: Float[Array, " points ..."], 
                               region_count: int,
                               max_steps: int
      ) -> Float[Array, " dim1 2 region_count"]:
  """Computes quantized credible regions based on a set of credible points.
  
  Applies k-means clustering to quantize the space of points. Computes the
  bounding boxes of clusters representing the quantized credible regions.
  
  Args:
    points: A set of points
    region_count: Number of credible regions
  
  Returns:
    A set of bounding boxes of the credible regions.
  """
  # Step 1, Apply k-means to quantize points
  initial_centers = points[:region_count,...].copy()
  centers, labels = kmeans(points, initial_centers, max_steps)

  # Step 2, draw bounding boxes around points
  out = []
  for i in range(region_count):
     cluster = centers[labels==i, ...]
     box = compute_bounding_box(cluster)
     out.append(box)

  return out


@jax.jit
def compute_bounding_box(points: Float[Array, " sample ..."]):
  """Computes a bounding box around a set of points"""
  upper = jnp.max(points, axis=0)
  lower = jnp.min(points, axis=0)
  return jnp.stack([upper, lower])


@jax.jit
def assign_center(point: Float[Array, " dim"], 
                  centers: Float[Array, " k dim"]
    ) -> int:
  """Returns the center for kmeans clustering.
  
  Args:
    point: The vector being assigned.
    centers: Set of centers to compare.
  
  Returns:
    The index of the nearest center to the given point. 
  """
  distances = jnp.sum(jnp.square(point[None, :] - centers), axis=1)
  return jnp.argmin(distances)
    

@jax.jit
def compute_centers(points: Float[Array, " sample dim"],
                    labels: Int[Array, " sample"],
                    old_centers: Float[Array, " k dim"]
) -> Float[Array, " k dim"]:
  """Given a labeled dataset, compute the centers.
  
  Args:
    points: A set of sample points.
    labels: labels for the sample points.
    old_centers: An array containing the previous centers. 

  Returns:
    An array of averages of the labeled points.

  Notes:
    old_centers is only used for the shape due to the requirement
    in Jax for type concretization.
    
  """
  k = old_centers.shape[0]
  sample_count = points.shape[0]

  @jax.jit
  def add_sample_to_sum(i, args):
    cluster_sum, label_count = args
    label = labels[i]

    label_count = label_count.at[label].add(1)
    cluster_sum = cluster_sum.at[label,:].add(points[i, :])
    return cluster_sum, label_count

  cluster_sum = jnp.zeros_like(old_centers)
  label_count = jnp.zeros(k, int)
  cluster_sum, label_count = jax.lax.fori_loop(0, 
                                               sample_count, 
                                               add_sample_to_sum, 
                                               (cluster_sum, label_count))

  new_centers = cluster_sum / label_count[:, None]
  return new_centers


@jax.jit
def kmeans(points: Float[Array, " point dim"],
           initial_centers: Float[Array, " k dim"],
           max_steps
    ) -> tuple[Float[Array, " k dim"], Int[Array, " point"]]:
  """Performs K-means clustering on an array.
  
  Args:
    points: An N by D array of N datapoints
    initial_centers: A K by D array of K initial centers
    max_steps: Maximum number of steps for k-means
  
  Returns:
    `(centers, labels)`
    where `centers` is a K by D array containing the means in k-means
    and `labels` is an N dimensional array mapping `points` onto
    their nearest center.
  """

  converged = False
  assign_center_v = jax.vmap(assign_center, (0, None))

  def kmeans_iteration(args):
    converged, old_centers, old_labels, i = args
    i += 1
    new_labels = assign_center_v(points, old_centers)
    new_centers = compute_centers(points, labels, old_centers)
    converged = (new_labels == old_labels).all()
    return converged, new_centers, new_labels, i

  def kmeans_condition(args):
    converged, _, _, i = args
    return converged | (i > max_steps)
  
  labels = jnp.zeros(points.shape[0], int)
  initial_args = converged, initial_centers, labels, 0

  out = jax.lax.while_loop(kmeans_condition, kmeans_iteration, initial_args)
  _, centers, labels, _ = out
  return centers, labels
