from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt, PIDController
import jax.numpy as jnp
from trajectory_manifold.examples import LorenzVectorField

vector_field = LorenzVectorField(5,8)

term = ODETerm(vector_field)
solver = Tsit5()
saveat = SaveAt(ts=[i*.1 for i in range(20)])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

y0 = jnp.ones(5)

sol = diffeqsolve(term, solver, t0=0, t1=5, dt0=0.1, y0=y0, saveat=saveat,
                  stepsize_controller=stepsize_controller) 

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])

