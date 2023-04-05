from jax.numpy import zeros
from jax.lax import fori_loop
from jax import jit
from typing import Callable
from jaxtyping import Array, Float, jaxtyped

@jaxtyped
def LorenzVectorField(N: int, F: float) \
        -> Callable[[Float, Float[Array, " dim"], any], Float[Array, " dim"]]:
    """ Returns a function representing the lorenz 96 system
    
    """
    @jit
    def out(t, y, args):
        d = zeros(y.shape[0])

        def inner(i, val):
            out, input = val
            N = input.shape[0]

            new_value = (input[i+1 % N] - input[i-2]) * input[i-1] - input[i] + F
            out = out.at[i].set(new_value)
            return out, input

        d, y = fori_loop(0, y.shape[0], inner, (d, y))

        return d

    return out

