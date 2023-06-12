from trajectory_manifold.examples import linear_vector_field
from trajectory_manifold.manifold import SolverParameters, system_pushforward_weight
from diffrax import PIDController, Heun

from trajectory_manifold.estimation import trajectory_likelihood
from trajectory_manifold.estimation import trajectory_log_likelihood
from trajectory_manifold.estimation import trajectory_log_posterior
from trajectory_manifold.estimation import trajectory_posterior
from trajectory_manifold.estimation import state_log_posterior
from trajectory_manifold.estimation import state_posterior

import jax.numpy as jnp

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

