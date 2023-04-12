from trajectory_manifold import trapezoidal_inner_product
from trajectory_manifold import trapezoidal_correlation
from trajectory_manifold import SolverParameters
from trajectory_manifold import system_sensitivity
from trajectory_manifold import system_pushforward_weight
from trajectory_manifold.examples import linear_vector_field

from trajectory_manifold.helpers import apply_kernel_vec

from jax.numpy import asarray, expand_dims
import jax.numpy as jnp
from diffrax import Tsit5

from math import exp, sqrt


class Test_trapezoidal_inner_product:
    def test_trapezoidal_norm(self):
        """Tests inner product of x**2 with itself"""
        x = asarray([(i/1000.0)**2 for i in range(1001)])
        x = expand_dims(x, 1)
        stepsize = 1/1000.0
        val = trapezoidal_inner_product(x, x, stepsize)
        assert abs(val - 1/5) < 0.001


    def test_trapezoidal_inner_product(self):
        """Tests inner product of x**2 with x"""
        x = asarray([(i/1000.0)**2 for i in range(1001)])
        y = asarray([(i/1000.0) for i in range(1001)])
        x = expand_dims(x, 1)
        y = expand_dims(y, 1)
        stepsize = 1/1000.0
        val = trapezoidal_inner_product(x, y, stepsize)
        assert abs(val - 1/4) < 0.001


class Test_trapezoidal_correlation:
    def test_trapezoidal_correlation(self):
        data = [[(i/1000.0)    for i in range(1001)],
                [(i/1000.0)**2 for i in range(1001)]]
        U = asarray(data)
        U = expand_dims(U, 2)

        stepsize = 1/1000.0
        results = trapezoidal_correlation(U, stepsize)
        assert abs(results[0,0] - 1/3) < 0.001
        assert abs(results[0,1] - 1/4) < 0.001
        assert abs(results[1,0] - 1/4) < 0.001
        assert abs(results[1,1] - 1/5) < 0.001


class Test_system_sensitivity:
    params = SolverParameters(solver=Tsit5(),
                              relative_tolerance=1e-5,
                              absolute_tolerance=1e-5,
                              time_horizon=1,
                              step_size=0.1)

    def test_dimensions(self):
        dynamics = asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        initial_condition = asarray([1.0, 1.0])
        U = system_sensitivity(vector_field, initial_condition, self.params)
        assert U.shape == (2, 11, 2)

    def test_dimension_order(self):
        dynamics = asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        initial_condition = asarray([1.0, 1.0])
        U = system_sensitivity(vector_field, initial_condition, self.params)
        print(U)
        assert abs(U[1,-1,1] - exp(1)) < 0.01
        assert abs(U[0,-1,0] - exp(-1)) < 0.01
        assert abs(U[0,-1,1]) < 0.01
        assert abs(U[1,-1,0]) < 0.01


class Test_system_pushforward_weight:
    def test_linear(self):
        dynamics = asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        time_horizon=1
        initial_condition = asarray([1.0, 1.0])
        weight = system_pushforward_weight(vector_field,
                                           time_horizon,
                                           initial_condition)
        truth = sqrt(0.25 * (1 - exp(-2)) * (exp(2) - 1))
        assert abs(weight - truth) < 0.1


class Test_apply_kernel_vec:
    def test_basic(self):
        functions = 2
        time_steps = 10
        dimensions = 3
        x = jnp.ones((functions, time_steps, dimensions))
        kernel = jnp.ones((time_steps, dimensions, dimensions))

        out = apply_kernel_vec(x, kernel)
        truth = 3 * jnp.ones((functions, time_steps, dimensions))
        assert (out == truth).all()
        
