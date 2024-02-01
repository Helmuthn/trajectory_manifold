from trajectory_manifold.examples import linear_vector_field
from trajectory_manifold.manifold import SolverParameters, system_pushforward_weight
from diffrax import PIDController, Heun

from trajectory_manifold.estimation import trajectory_likelihood
from trajectory_manifold.estimation import trajectory_log_likelihood
from trajectory_manifold.estimation import trajectory_log_posterior
from trajectory_manifold.estimation import trajectory_posterior
from trajectory_manifold.estimation import state_log_posterior
from trajectory_manifold.estimation import state_posterior
from trajectory_manifold.estimation import assign_center
from trajectory_manifold.estimation import compute_centers
from trajectory_manifold.estimation import kmeans
from trajectory_manifold.estimation import compute_bounding_box

import jax.numpy as jnp
import jax.random as random

def gaussian_log_likelihood(obs, state):
    return jnp.sum(jnp.square(obs - state))/2 - jnp.log(jnp.sqrt(2*jnp.pi))

def gaussian_likelihood(obs, state):
    return jnp.exp(jnp.sum(jnp.square(obs - state))/2)/(jnp.sqrt(2*jnp.pi))

def uniform_prior(state, parameters):
    return 1.0/2.0

def uniform_log_prior(state, parameters):
    return -jnp.log(2.0)


class Test_trajectory_log_likelihood:

    def test_linear_sys(self):
        A = -1 * jnp.ones((1,1))
        vector_field = linear_vector_field(A)

        observations = jnp.ones((2,1))
        observation_times = jnp.arange(2)
        parameters = SolverParameters(stepsize_controller = PIDController(rtol=1e-3, atol=1e-3),
                                      step_size_internal=1e-2,
                                      step_size_output=1.0,
                                      time_interval=(0.0,1.1),
                                      solver=Heun(),
                                      max_steps=int(1e5))

        def true_ll(state, system_params):
            return gaussian_log_likelihood(state, jnp.ones(1)) \
                 + gaussian_log_likelihood(state * jnp.exp(-1), jnp.ones(1))

        test_ll = trajectory_log_likelihood(vector_field,
                                            observations,
                                            observation_times,
                                            gaussian_log_likelihood,
                                            parameters)

        assert jnp.abs(true_ll(jnp.ones(1), jnp.zeros(0)) - test_ll(jnp.ones(1), jnp.zeros(0))) < 0.05
    
    

class Test_trajectory_likelihood:
    def test_linear_sys(self):
        A = -1 * jnp.ones((1,1))
        vector_field = linear_vector_field(A)

        observations = jnp.ones((2,1))
        observation_times = jnp.arange(2)
        parameters = SolverParameters(stepsize_controller = PIDController(rtol=1e-3, atol=1e-3),
                                      step_size_internal=1e-2,
                                      step_size_output=1.0,
                                      time_interval=(0.0,1.1),
                                      solver=Heun(),
                                      max_steps=int(1e5))

        def true_l(state, system_params):
            return gaussian_likelihood(state, jnp.ones(1)) \
                 * gaussian_likelihood(state * jnp.exp(-1), jnp.ones(1))

        test_l = trajectory_likelihood(vector_field,
                                        observations,
                                        observation_times,
                                        gaussian_likelihood,
                                        parameters)

        assert jnp.abs(true_l(jnp.ones(1), jnp.zeros(0)) - test_l(jnp.ones(1), jnp.zeros(0))) < 0.05

class Test_state_log_posterior:

    def test_linear_sys(self):
        A = -1 * jnp.ones((1,1))
        vector_field = linear_vector_field(A)

        observations = jnp.ones((2,1))
        observation_times = jnp.arange(2)
        parameters = SolverParameters(stepsize_controller = PIDController(rtol=1e-3, atol=1e-3),
                                      step_size_internal=1e-2,
                                      step_size_output=1.0,
                                      time_interval=(0.0,1.1),
                                      solver=Heun(),
                                      max_steps=int(1e5))

        def true_lp(state, system_params):
            return gaussian_log_likelihood(state, jnp.ones(1)) \
                 + gaussian_log_likelihood(state * jnp.exp(-1), jnp.ones(1))\
                 + uniform_log_prior(state, system_params)

        test_ll = state_log_posterior(vector_field,
                                           observations,
                                           observation_times,
                                           gaussian_log_likelihood,
                                           uniform_log_prior,
                                           parameters)

        assert jnp.abs(true_lp(jnp.ones(1), jnp.zeros(0)) - test_ll(jnp.ones(1), jnp.zeros(0))) < 0.05

class Test_state_posterior:
    
    def test_linear_sys(self):
        A = -1 * jnp.ones((1,1))
        vector_field = linear_vector_field(A)

        observations = jnp.ones((2,1))
        observation_times = jnp.arange(2)
        parameters = SolverParameters(stepsize_controller = PIDController(rtol=1e-3, atol=1e-3),
                                      step_size_internal=1e-2,
                                      step_size_output=1.0,
                                      time_interval=(0.0,1.1),
                                      solver=Heun(),
                                      max_steps=int(1e5))

        def true_lp(state, system_params):
            return gaussian_likelihood(state, jnp.ones(1)) \
                 * gaussian_likelihood(state * jnp.exp(-1), jnp.ones(1))\
                 * uniform_prior(state, system_params)

        test_ll = state_posterior(vector_field,
                                  observations,
                                  observation_times,
                                  gaussian_likelihood,
                                  uniform_prior,
                                  parameters)

        assert jnp.abs(true_lp(jnp.ones(1), jnp.zeros(0)) - test_ll(jnp.ones(1), jnp.zeros(0))) < 0.05

class Test_trajectory_log_posterior:

    def test_linear_sys(self):
        A = -1 * jnp.ones((1,1))
        vector_field = linear_vector_field(A)

        observations = jnp.ones((2,1))
        observation_times = jnp.arange(2)
        parameters = SolverParameters(stepsize_controller = PIDController(rtol=1e-3, atol=1e-3),
                                      step_size_internal=1e-2,
                                      step_size_output=1.0,
                                      time_interval=(0.0,1.1),
                                      solver=Heun(),
                                      max_steps=int(1e5))
        weight = system_pushforward_weight(vector_field,
                                           (0.0, 1.0),
                                           jnp.ones(1),
                                           jnp.ones(0))

        def true_lp(state, system_params):
            return gaussian_log_likelihood(state, jnp.ones(1)) \
                 + gaussian_log_likelihood(state * jnp.exp(-1), jnp.ones(1))\
                 + uniform_log_prior(state, system_params) \
                 - jnp.log(weight)

        test_lp = trajectory_log_posterior(vector_field,
                                           observations,
                                           observation_times,
                                           gaussian_log_likelihood,
                                           uniform_log_prior,
                                           (0.0, 1.01),
                                           parameters)

        assert jnp.abs(true_lp(jnp.ones(1), jnp.zeros(0)) - test_lp(jnp.ones(1), jnp.zeros(0))) < 0.05

class Test_trajectory_posterior:

    def test_linear_sys(self):
        A = -1 * jnp.ones((1,1))
        vector_field = linear_vector_field(A)

        observations = jnp.ones((2,1))
        observation_times = jnp.arange(2)
        parameters = SolverParameters(stepsize_controller = PIDController(rtol=1e-3, atol=1e-3),
                                      step_size_internal=1e-2,
                                      step_size_output=1.0,
                                      time_interval=(0.0,1.1),
                                      solver=Heun(),
                                      max_steps=int(1e5))

        weight = system_pushforward_weight(vector_field,
                                           (0.0, 1.0),
                                           jnp.ones(1),
                                           jnp.ones(0))

        def true_p(state, system_params):
            return gaussian_likelihood(state, jnp.ones(1)) \
                 * gaussian_likelihood(state * jnp.exp(-1), jnp.ones(1))\
                 * uniform_prior(state, system_params) \
                 / weight

        test_p = trajectory_posterior(vector_field,
                                          observations,
                                          observation_times,
                                          gaussian_likelihood,
                                          uniform_prior,
                                          (0.0, 1.01),
                                          parameters)

        assert jnp.abs(true_p(jnp.ones(1), jnp.zeros(0)) - test_p(jnp.ones(1), jnp.zeros(0))) < 0.05


class Test_assign_center:
    def test_main_behavior(self):
        centers = jnp.ones((2,5))
        centers = centers.at[1,:].set(0)
        point1 = jnp.zeros(5) + 0.1
        point2 = jnp.ones(5) + 0.1

        assert assign_center(point1, centers) == 1
        assert assign_center(point2, centers) == 0

    def test_main_behavior_reversed(self):
        centers = jnp.ones((2,5))
        centers = centers.at[0,:].set(0)
        point1 = jnp.zeros(5) + 0.1
        point2 = jnp.ones(5) + 0.1

        assert assign_center(point1, centers) == 0
        assert assign_center(point2, centers) == 1


class Test_compute_centers:
    def test_typical_behavior(self):
        points = jnp.ones((10, 2))
        labels = jnp.ones(10, int)
        old_centers = jnp.ones((2, 2))
        labels = labels.at[0:5].set(0)
        points = points.at[0:5,:].set(2)

        new_centers = compute_centers(points, labels, old_centers)
        assert new_centers[0,0] == 2
        assert new_centers[0,1] == 2
        assert new_centers[1,0] == 1
        assert new_centers[1,1] == 1

class Test_kmeans:
    def test_typical_behavior(self):
        key = random.PRNGKey(1234)
        points = jnp.ones((10, 2))
        labels = jnp.ones(10, int)
        labels = labels.at[0:5].set(0)
        points = points.at[0:5,:].set(2)
        initial_centers = random.normal(key, (2,2))
        centers, labels = kmeans(points, initial_centers, 100)

        assert (labels[0:4] == labels[0]).all()
        assert (labels[5:] == labels[5]).all()

class Test_compute_bounding_box:
    def test_output_shape(self):
        points = jnp.ones((3,3,4))
        points = points.at[0,:,:].set(0)
        out = compute_bounding_box(points)
        assert out.shape == (2,3,4)

    def test_basic_behavior(self):
        points = jnp.ones((3,3,4))
        points = points.at[0,:,:].set(0)
        out = compute_bounding_box(points)
        assert (out[0,...] == 1).all()
        assert (out[1,...] == 0).all()

