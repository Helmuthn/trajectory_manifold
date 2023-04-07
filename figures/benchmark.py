import sys
import os
sys.path.insert(0, os.path.abspath('.'))
print(os.path.abspath('.'))

from trajectory_manifold.examples import LinearVectorField
from trajectory_manifold.main import system_pushforward_weight
import jax.numpy as jnp
from jax import jit
import timeit


vector_field = LinearVectorField(jnp.ones((2,2)))
initial_condition = jnp.ones(2)

@jit
def testfunc(x):
    return system_pushforward_weight(vector_field, 1, x)

start_time = timeit.default_timer()
for i in range(1000):
    testfunc(initial_condition)
end_time = timeit.default_timer()
print(str(end_time - start_time) + " ms per function call")

