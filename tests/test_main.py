from trajectory_manifold.main import trapezoidal_inner_product
from trajectory_manifold.main import trapezoidal_correlation
from trajectory_manifold.main import SolverParameters
from trajectory_manifold.main import system_sensitivity
from trajectory_manifold.examples import LinearVectorField

from jax.numpy import asarray, expand_dims
from diffrax import Tsit5

from math import exp


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
        vector_field = LinearVectorField(dynamics)
        initial_condition = asarray([1.0, 1.0])
        U = system_sensitivity(vector_field, initial_condition, self.params)
        assert U.shape == (2, 11, 2)

    def test_dimension_order(self):
        dynamics = asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = LinearVectorField(dynamics)
        initial_condition = asarray([1.0, 1.0])
        U = system_sensitivity(vector_field, initial_condition, self.params)
        print(U)
        assert abs(U[1,-1,1] - exp(1)) < 0.01
        assert abs(U[0,-1,0] - exp(-1)) < 0.01
        assert abs(U[0,-1,1]) < 0.01
        assert abs(U[1,-1,0]) < 0.01
