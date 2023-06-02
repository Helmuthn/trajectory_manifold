from trajectory_manifold.manifold import SolverParameters
from trajectory_manifold.manifold import system_sensitivity
from trajectory_manifold.manifold import system_pushforward_weight
from trajectory_manifold.manifold import system_sensitivity_and_solution
from trajectory_manifold.examples import linear_vector_field


import jax.numpy as jnp
from diffrax import Heun, ConstantStepSize

from math import exp, sqrt



class Test_system_sensitivity:
    params = SolverParameters(solver=Heun(),
                              time_interval=(0.0,1.01),
                              step_size_output=0.1,
                              step_size_internal=1e-1,
                              stepsize_controller=ConstantStepSize(),
                              max_steps=int(1e2))
    system_parameters = jnp.zeros(0)

    def test_dimensions(self):
        dynamics = jnp.asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        initial_condition = jnp.asarray([1.0, 1.0])
        U = system_sensitivity(vector_field, 
                               initial_condition, 
                               self.system_parameters, 
                               self.params)

        assert U[0].shape == (2, 11, 2)
        assert U[1].shape == (0, 11, 2)

    def test_dimension_order(self):
        dynamics = jnp.asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        initial_condition = jnp.asarray([1.0, 1.0])
        U = system_sensitivity(vector_field, 
                               initial_condition, 
                               self.system_parameters,
                               self.params)[0]

        assert abs(U[1,-1,1] - exp(1)) < 0.01
        assert abs(U[0,-1,0] - exp(-1)) < 0.01
        assert abs(U[0,-1,1]) < 0.01
        assert abs(U[1,-1,0]) < 0.01


class Test_system_pushforward_weight:
    system_parameters = jnp.zeros(0)

    def test_linear(self):
        dynamics = jnp.asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        time_interval=(0.0,1.0)
        initial_condition = jnp.asarray([1.0, 1.0])
        weight = system_pushforward_weight(vector_field,
                                           time_interval,
                                           initial_condition,
                                           self.system_parameters)

        truth = sqrt(0.25 * (1 - exp(-2)) * (exp(2) - 1))
        assert abs(weight - truth) < 0.1

class Test_system_sensitivity_and_solution:
    system_parameters = jnp.zeros(0)
    params = SolverParameters(solver=Heun(),
                              time_interval=(0.0,1.01),
                              step_size_output=0.1,
                              step_size_internal=1e-1,
                              stepsize_controller=ConstantStepSize(),
                              max_steps=int(1e2))

    def test_shape(self):
        dynamics = jnp.asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        initial_condition = jnp.asarray([1.0, 1.0])
        U, sol = system_sensitivity_and_solution(vector_field, 
                                                 initial_condition, 
                                                 self.system_parameters, 
                                                 self.params)
        assert U[0].shape == (2, 11, 2)
        assert U[1].shape == (0, 11, 2)
        assert sol.shape == (11, 2)

