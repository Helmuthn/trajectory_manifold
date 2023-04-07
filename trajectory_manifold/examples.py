from jax.numpy import zeros
from jax.lax import fori_loop
from jax import jit
from typing import Callable
from jaxtyping import Array, Float, jaxtyped

@jaxtyped
def LorenzVectorField(
    F: float
) -> Callable[[Float, Float[Array, " dim"], any], Float[Array, " dim"]]:
    """ Returns a function representing the Lorenz 96 system.
    
    Args:
        F: The constant forcing term
    
    Returns:
        A function representing the vector field for the Lorenz 96 system.
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


@jaxtyped
def LinearVectorField(
    A: Float[Array, " dim dim"]
) -> Callable[[Float, Float[Array, " dim"], any], Float[Array, " dim"]]:
    """ Returns a function representing a linear system.
    
    Args:
        A: A matrix defining the system dynamics
    
    Returns:
        A function representing teh vector field for a linear system.
    """

    @jit
    @jaxtyped
    def out(t: any, 
            y: Float[Array, " dim"], 
            args: any,
    ) -> Float[Array, " dim"]:
        return A @ y

    return out