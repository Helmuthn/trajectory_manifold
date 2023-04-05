from trajectory_manifold.main import trapezoidal_inner_product
from trajectory_manifold.main import trapezoidal_correlation
from jax.numpy import asarray


class Test_trapezoidal_inner_product:
    def test_trapezoidal_norm(self):
        """Tests inner product of x**2 with itself"""
        x = asarray([(i/1000.0)**2 for i in range(1001)])
        stepsize = 1/1000.0
        val = trapezoidal_inner_product(x, x, stepsize)
        assert abs(val - 1/5) < 0.001


    def test_trapezoidal_inner_product(self):
        """Tests inner product of x**2 with x"""
        x = asarray([(i/1000.0)**2 for i in range(1001)])
        y = asarray([(i/1000.0) for i in range(1001)])
        stepsize = 1/1000.0
        val = trapezoidal_inner_product(x, y, stepsize)
        assert abs(val - 1/4) < 0.001


class Test_trapezoidal_correlation:
    def test_trapezoidal_correlation(self):
        data = [[(i/1000.0)    for i in range(1001)],
                [(i/1000.0)**2 for i in range(1001)]]
        U = asarray(data)
        stepsize = 1/1000.0
        results = trapezoidal_correlation(U, stepsize)
        assert abs(results[0,0] - 1/3) < 0.001
        assert abs(results[0,1] - 1/4) < 0.001
        assert abs(results[1,0] - 1/4) < 0.001
        assert abs(results[1,1] - 1/5) < 0.001