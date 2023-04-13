from trajectory_manifold.helpers import trapezoidal_inner_product
from trajectory_manifold.helpers import trapezoidal_correlation
from trajectory_manifold.helpers import apply_kernel_vec

import jax.numpy as jnp

class Test_trapezoidal_inner_product:
    def test_trapezoidal_norm(self):
        """Tests inner product of x**2 with itself"""
        x = jnp.asarray([(i/1000.0)**2 for i in range(1001)])
        x = jnp.expand_dims(x, 1)
        stepsize = 1/1000.0
        val = trapezoidal_inner_product(x, x, stepsize)
        assert abs(val - 1/5) < 0.001


    def test_trapezoidal_inner_product(self):
        """Tests inner product of x**2 with x"""
        x = jnp.asarray([(i/1000.0)**2 for i in range(1001)])
        y = jnp.asarray([(i/1000.0) for i in range(1001)])
        x = jnp.expand_dims(x, 1)
        y = jnp.expand_dims(y, 1)
        stepsize = 1/1000.0
        val = trapezoidal_inner_product(x, y, stepsize)
        assert abs(val - 1/4) < 0.001


class Test_trapezoidal_correlation:
    def test_trapezoidal_correlation(self):
        data = [[(i/1000.0)    for i in range(1001)],
                [(i/1000.0)**2 for i in range(1001)]]
        U = jnp.asarray(data)
        U = jnp.expand_dims(U, 2)

        stepsize = 1/1000.0
        results = trapezoidal_correlation(U, stepsize)
        assert abs(results[0,0] - 1/3) < 0.001
        assert abs(results[0,1] - 1/4) < 0.001
        assert abs(results[1,0] - 1/4) < 0.001
        assert abs(results[1,1] - 1/5) < 0.001


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
        