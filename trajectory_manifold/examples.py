import jax.numpy as jnp
from jax.lax import fori_loop
from jax import jit
from typing import Callable
from jaxtyping import Array, Float, jaxtyped, PyTree


@jaxtyped
def lorenz_vector_field(
    sigma: float = 10,
    rho: float = 28,
    beta: float = 8.0/3.0
) -> Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]]:
    """ Returns a function representing the Lorenz system.
    
    Args:
        sigma: Lorenz Parameter
        rho: Lorenz Parameter
        beta: Lorenz Parameter
    
    Returns:
        A function representing the vector field for the Lorenz 96 system.
    """
    @jit
    def out(t, y, args):
        d = jnp.zeros(3)

        d = d.at[0].set(sigma * (y[1] - y[0]))
        d = d.at[1].set(y[0] * (rho - y[2]) - y[1])
        d = d.at[2].set(y[0] * y[1] - beta * y[2])

        return d

    return out


@jaxtyped
def lorenz96_vector_field(
    F: float
) -> Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]]:
    """ Returns a function representing the Lorenz 96 system.
    
    Args:
        F: The constant forcing term
    
    Returns:
        A function representing the vector field for the Lorenz 96 system.
    """
    @jit
    def out(t, y, args):
        d = jnp.zeros(y.shape[0])

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
def linear_vector_field(
    A: Float[Array, " dim dim"]
) -> Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]]:
    """ Returns a function representing a linear system.
    
    Args:
        A: A matrix defining the system dynamics
    
    Returns:
        A function representing the vector field for a linear system.
    """

    @jit
    @jaxtyped
    def out(t: Float, 
            y: Float[Array, " dim"], 
            args: PyTree,
    ) -> Float[Array, " dim"]:
        return A @ y

    return out


@jaxtyped
def lotka_volterra_vector_field(
    prey_growth: float,
    predation_rate: float,
    predator_growth: float,
    predator_decay: float,

) -> Callable[[Float, Float[Array, " dim"], PyTree], Float[Array, " dim"]]:
    """ Returns a function representing a Lotka-Volterra system.
    
    Args:
        prey_growth: Exponential growth rate of prey without predation.
        predation_rate: Decay rate for prey in contact with predator.
        predator_growth: Growth rate for predators in contact with prey.
        predator_decay: Exponential decay rate of predators without prey.
    
    Returns:
        A function representing the vector field for a Lotka-Volterra system.
    """

    @jit
    @jaxtyped
    def out(t: Float, 
            y: Float[Array, " dim"], 
            args: PyTree,
    ) -> Float[Array, " dim"]:

        prey, predators = y

        prey_rate  = prey_growth * prey 
        prey_rate -= predation_rate * prey * predators

        predator_rate  = predator_growth * prey * predators 
        predator_rate -= predator_decay * predators

        return jnp.asarray([prey_rate, predator_rate])

    return out