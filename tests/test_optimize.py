import jax.numpy as jnp
from trajectory_manifold.optimize import grid_optimum
from trajectory_manifold.optimize import distance_gradient
from trajectory_manifold.manifold import SolverParameters
from trajectory_manifold.examples import linear_vector_field
from diffrax import Heun, ConstantStepSize

class Test_grid_optimum:
    x = jnp.arange(0,2,0.5)
    y = jnp.arange(0,2,0.5)

    grid = jnp.zeros((x.shape[0], y.shape[0]))
    grid = grid.at[1,1].set(1)
    grid = grid.at[2,1].set(-1)

    def test_max(self):
        x = self.x 
        y = self.y 
        grid = self.grid

        truth = jnp.stack([x[1], y[1]])
        result = grid_optimum(grid, x, y, True)
        assert (truth == result).all()

    def test_min(self):
        x = self.x 
        y = self.y 
        grid = self.grid

        truth = jnp.stack([x[2], y[1]])
        result = grid_optimum(grid, x, y, False)
        assert (truth == result).all()

    def test_reverse(self):
        x = self.x 
        y = self.y 
        grid = self.grid

        truth = jnp.stack([y[1], x[2]])
        result = grid_optimum(grid, x, y, False, True)
        assert (truth == result).all()

class Test_distance_gradient:
    params = SolverParameters(solver=Heun(),
                              time_interval=(0.0,1.01),
                              step_size_output=0.1,
                              step_size_internal=1e-1,
                              stepsize_controller=ConstantStepSize(),
                              max_steps=int(1e2))
    system_parameters = jnp.zeros(0)

    def test_shape(self):
        dynamics = jnp.asarray([[-1.0, 0.0],[0.0, 1.0]])
        vector_field = linear_vector_field(dynamics)
        initial_condition = jnp.asarray([1.0, 1.0])
        trajectory = jnp.zeros((11,2))
        U = distance_gradient(initial_condition, 
                              self.system_parameters, 
                              vector_field,
                              trajectory,
                              self.params)

        assert U[0].shape == (1, 2)
        assert U[1].shape == (1, 0)