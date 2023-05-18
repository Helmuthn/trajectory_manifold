import jax.numpy as jnp
from trajectory_manifold.optimize import grid_optimum

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