"""This module contains linear algebra helper functions.

The purpose of this module is to adapt standard linear algebra
fuctions to the needs of this project. These include items such
as computation of an inner product through the trapezoidal rule.
"""

from jax import jit, vmap, tree_map
from jax.lax import fori_loop, cond
import jax.numpy as jnp


from jaxtyping import Float, Array, PyTree

@jit
def frobenius_inner_product(
    x: Float[Array, " *dim"], 
    y: Float[Array, " *dim"], 
) -> Float:
    """Computes the Frobenius inner product of two matrices.

    Given two multidimensional arrays, computes the sum of
    the elementwise product of the arrays.
    
    Args:
        x: A multidimensional array.
        y: A multidimensional array.
    
    Returns:
        The sum of the elementwise product of x and y.
    """
    return jnp.sum(x * y)


@jit
def trapezoidal_inner_product(
    x: Float[Array, " timesteps dim"], 
    y: Float[Array, " timesteps dim"], 
    step_size: Float,
) -> Float:
    """Approximate the inner product by the trapezoidal rule.
    
    Computes an approximate inner product between two functions represented
    by a finite-grid approximation with a fixed step size.
    Computed through the trapezoidal integration scheme.
    
    Args:
        x: Grid approximation of the first function. Each row represents the 
          value of the multivariate function at a given timestep.
        y: Grid approximation of the second function. Each row represents the
          value of the multivariate function at a given timestep.
        step_size: Spacing between sample points in the functions.
        
    Returns:
        An approximation of the L2 inner product.
    """

    out = (jnp.dot(x[0, :],  y[0, :]) + jnp.dot(x[-1, :], y[-1, :])) / 2
    out += frobenius_inner_product(x[1:-1, :], y[1:-1, :])
    return out * step_size
    

def trapezoidal_pytree_vector_product(
        y: Float[Array, " timesteps dim"],
        X: PyTree,
        step_size: Float,
    ) -> PyTree:
    """Computes the pytree-vector product where leaves are sampled functions.
    
    Given a PyTree where leaves are 2-dimensional arrays representing
    sampled function and another 2-dimensional array reepresenting
    a single multivariate function, returns a PyTree of the same shape
    as the original input with leaves resulting from the L2 inner
    products with the function.
    
    Args:
        y: A 2D array with indices (timestep, dim)
        X: A PyTree where leaves are 2D arrays with indices (timestep, dim)
        step_size: Step size required for numerical integration scaling
    
    Returns:
        A PyTree of the same shape a `X`, but with results of inner products
        as leaves.
    """

    @jit
    def is_leaf(x):
        if isinstance(x, jnp.ndarray) \
           and len(x.shape) == 2 \
           and jnp.issubdtype(x.dtype, jnp.number):
            return True
        else:
            return False

    @jit
    def apply_vector(x):
        return trapezoidal_inner_product(x, y, step_size)

    return tree_map(apply_vector, X, is_leaf=is_leaf)

@jit
def trapezoidal_matrix_product(
    X: Float[Array, " functions1 timesteps dim"],
    Y: Float[Array, " functions2 timesteps dim"],
    step_size: Float,
) -> Float[Array, " functions1 functions2"]:
    """Computes the product of two matrices where rows represent functions.
    
    Given two matrices, `X` and `Y`, such that the first index indicates
    a chosen function, computes `X @ Y.T` based on the trapezoidal rule for 
    integration.
    
    Args:
        X: An M by N by K array of N samples of each of M functions
          taking values in a K-dimensional space. 
        Y: An M by N by K array of N samples of each of M functions
          taking values in a K-dimensional space. 
        step_size: Spacing between sample points in the functions.

    Returns:
        The result of the generalized matrix-matrix product.
    """

    vv = lambda x, y: trapezoidal_inner_product(x, y, step_size)
    mv = vmap(vv, (0, None), 0)
    mm = vmap(mv, (None, 0), 1)
    return mm(X, Y)
    

@jit
def trapezoidal_inner_product_weighted(
    x: Float[Array, " timesteps dim"], 
    y: Float[Array, " timesteps dim"], 
    step_size: Float,
    kernel: Float[Array, " timesteps dim dim"]
) -> Float:
    """Approximate the inner product by the trapezoidal rule.
    
    Computes an approximate inner product between two functions represented
    by a finite-grid approximation with a fixed step size.
    Computed through the trapezoidal integration scheme.
    
    Args:
        x: Grid approximation of the first function. Each row represents the 
          value of the multivariate function at a given timestep.
        y: Grid approximation of the second function. Each row represents the
          value of the multivariate function at a given timestep.
        step_size: Spacing between sample points in the functions.
        
    Returns:
        An approximation of the L2 inner product.
    """
    x = apply_kernel(x, kernel)

    return trapezoidal_inner_product(x, y, step_size)


@jit
def apply_kernel(
    x: Float[Array, " timesteps dim"], 
    kernel: Float[Array, " timesteps dim dim"],
) -> Float[Array, " timesteps dim"]:
    """Helper Function to apply an integration kernel to an input.

    Applies the kernel to an input at each timestep.

    Args:
        x: An N by K array of N samples of each of a function
          taking values in a K-dimensional space.
        kernel: An N by K by K array of N samples of an integral kernel
          applying transformations to each associated timestep of x
    
    Returns:
        An N by K array represented the result of the transformation.
    """

    @jit
    def inner(i, state):
        return state.at[i,:].set(kernel[i,:,:] @ state[i,:])

    return fori_loop(0, x.shape[0], inner, x)


@jit
def apply_kernel_vec(
    x: Float[Array, " functions timesteps dim"], 
    kernel: Float[Array, " timesteps dim dim"],
) -> Float[Array, " timesteps dim"]:
    """Helper Function to apply an integration kernel to an input.

    Applies the kernel to an input at each timestep.
    Vectorized along a set of functions.

    Args:
        x: An M by N by K array of N samples of each of M functions
          taking values in a K-dimensional space.
        kernel: An N by K by K array of N samples of an integral kernel
          applying transformations to each associated timestep of x
    
    Returns:
        An M by N by K array represented the result of the transformation.
    """

    return vmap(apply_kernel, in_axes=(0, None))(x, kernel)

@jit
def trapezoidal_correlation(
    U: Float[Array, " functions timesteps dim"],
    step_size: Float,
) -> Float[Array, " functions functions"]:
    """Computes the inner products between rows of the given matrix.
    
    Constructs an M by M matrix of approximate inner products between M 
    multi-variate functions computed using N evenly spaced samples in a 
    trapezoidal integration scheme. 
    
    Args:
        U: M by N by K array representing N samples each of M functions that
          take values in a K-dimensional space.
        step_size: Spacing between sample points in the functions.
    
    Returns:
        An M by M matrix where the (i,j)'th element is the approximate
        inner product between rows i and j of the input matrix. 
    """

    M = U.shape[0]
    out = jnp.zeros((M, M))
    if M == 0:
        return out

    def inner(i, val):
        out, U, step_size = val

        def inner2(j, val):
            out, U, step_size, i = val
            value = trapezoidal_inner_product(U[i,...], U[j,...], step_size)
            out = out.at[i,j].set(value)
            out = out.at[j,i].set(value)
            return (out, U, step_size, i)

        out, U, step_size, i = fori_loop(i, M, inner2, (out, U, step_size, i))

        return out, U, step_size

    out, U, step_size = fori_loop(0, M, inner, (out, U, step_size))

    return out


@jit
def trapezoidal_correlation_weighted(
    U: Float[Array, " functions timesteps dim"],
    step_size: Float,
    kernel: Float[Array, " timesteps dim dim"],
) -> Float[Array, " functions functions"]:
    """Computes the inner products between rows of the given matrix.
    
    Constructs an M by M matrix of approximate inner products between M 
    multi-variate functions computed using N evenly spaced samples in a 
    trapezoidal integration scheme. 
    Uses an integral kernel to define the inner product.
    
    Args:
        U: M by N by K array representing N samples each of M functions that
          take values in a K-dimensional space.
        step_size: Spacing between sample points in the functions.
        kernel: N by K by K array representing an integral kernel defining
          the inner product in the underlying space.
    
    Returns:
        An M by M matrix where the (i,j)'th element is the approximate
        inner product between rows i and j of the input matrix. 
    """

    M = U.shape[0]
    out = jnp.zeros((M, M))
    U_reweight = apply_kernel_vec(U, kernel)
    
    @jit
    def inner(i, val):
        out, U, step_size = val

        @jit
        def inner2(j, val):
            out, U, step_size, i = val
            value = trapezoidal_inner_product(U[i,...], 
                                              U_reweight[j,...], 
                                              step_size)
            out = out.at[i,j].set(value)
            out = out.at[j,i].set(value)
            return (out, U, step_size, i)

        out, U, step_size, i = fori_loop(i, M, inner2, (out, U, step_size, i))

        return out, U, step_size

    out, U, step_size = fori_loop(0, M, inner, (out, U, step_size))

    return out

@jit
def safe_det(A: Float[Array, " dim dim"]):
    """Return the determinant with an empty matrix set to 1
    
    Used for computing the determinants of block diagonal matrices
    in the case where a block size may be zero
    
    Args:
        A: A matrix to compute the determinant
        
    Returns:
        The determinant of the matrix, or 1 if the matrix is empty
    """
    return cond(A.shape[0] == 0, lambda x: 1.0, jnp.linalg.det, A)