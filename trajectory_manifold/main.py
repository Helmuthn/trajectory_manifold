from jax import jit
from jax.lax import fori_loop
from jax.numpy import dot, zeros, sqrt
from jax.numpy.linalg import det
from jaxtyping import Float, Array
from typing import Callable

@jit
def trapezoidal_inner_product(
        x: Float[Array, " dim"], 
        y: Float[Array, " dim"], 
        stepsize: Float) \
        -> Float:
    """Approximate the inner product by the trapezoidal rule"""

    out = (x[0] * y[0] + x[-1] * y[-1]) / 2
    out += dot(x[1:-1], y[1:-1])
    return out * stepsize


@jit
def trapezoidal_correlation(
        U: Float[Array, " dim1 dim2"],
        stepsize: Float) \
        -> Float[Array, " dim1 dim1"]:
    """Computes the inner products between rows of the given matrix"""
    N = U.shape[0]
    out = zeros((N, N))

    def inner(i, val):
        out, U, stepsize = val

        def inner2(j, val):
            out, U, stepsize, i = val
            value = trapezoidal_inner_product(U[i,:], U[j,:], stepsize)
            out = out.at[i,j].set(value)
            out = out.at[j,i].set(value)
            return (out, U, stepsize, i)

        out, U, stepsize, i = fori_loop(i, N, inner2, (out, U, stepsize, i))

        return out, U, stepsize

    out, U, stepsize = fori_loop(0, N, inner, (out, U, stepsize))

    return out


def system_sensitivity(
        vector_field: Callable[[Float[Array, " dim"]], Float[Array, " dim"]], 
        time: Float, 
        initial_condition: Float[Array, " dim"]) \
        -> Float[Array, " dim"]:
    pass


def system_pushforward_weight(
        vector_field: Callable[[Float[Array, " dim"]], Float[Array, " dim"]], 
        time: Float, 
        initial_condition: Float[Array, " dim"]) \
        -> Float:
    """Constructs the pushforward weight for a given initial condition"""

    solver_tolerance = 1e-4
    step_size = 0.01
    U = system_sensitivity(vector_field, time, initial_condition)
    A = trapezoidal_correlation(U)
    return sqrt(abs(det(A)))