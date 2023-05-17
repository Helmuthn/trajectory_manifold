=====
Usage
=====

Installation
------------

This project is not yet published on pypi.

For now, clone the repository with::

    git clone https://github.com/Helmuthn/trajectory_manifold.git

then ``cd`` into the directory and install with::

    pip install .

This project depends on `jax 0.4.3+ <https://github.com/google/jax>`_, `diffrax 0.3.0+ <https://github.com/patrick-kidger/diffrax>`_, and `jaxtyping <https://github.com/google/jaxtyping>`_.

Note that jax installation is a bit more specialized and requires selection
dependent on your particular system. Thus, it is advised that you install it before this package.

Quick Start 
-----------

The key insight in this project is to interpret forecasting of ODE based 
systems as a reparameterization of the state estimation problem.

In this example, we will use the Lotka-Volterra system, provided in the
`examples` module of the library.
We begin by initializing the system and choosing the parameters for the 
ODE solvers.

.. code-block:: python

    from trajectory_manifold import examples
    from trajectory_manifold.manifold import SolverParameters

    vector_field = examples.lotka_volterra_vector_field(1,2,4,2)

    parameters = SolverParameters(relative_tolerance = 1e-2,
                                  absolute_tolerance = 1e-2,
                                  step_size = 0.1,
                                  time_interval = (0,10),
                                  solver=Heun(),
                                  max_steps=16**5)


We next simulate an observation process.
To do so, we use Jax for random number generation and Diffrax for
the ODE solvers.

.. code-block:: python

    dimension = 2
    noise_std = 1 

    key = random.PRNGKey(123)
    key, subkey = random.split(key)
    true_init = 2 * random.uniform(subkey, shape=(dimension,)) + center - 1

    term = ODETerm(vector_field)
    solver = parameters.solver
    observation_times = jnp.arange(parameters.time_interval[0], 
                                   parameters.time_interval[1] + parameters.step_size, 
                                   step=parameters.step_size)

    saveat = SaveAt(ts = observation_times)

    stepsize_controller = PIDController(rtol = parameters.relative_tolerance,
                                        atol = parameters.absolute_tolerance)

    states = diffeqsolve(term,
                         solver,
                         t0 = parameters.time_interval[0],
                         t1 = parameters.time_interval[1],
                         dt0 = parameters.step_size,
                         saveat = saveat,
                         stepsize_controller = stepsize_controller,
                         y0 = true_init).ys


    key, subkey = random.split(key)
    noise = noise_std*random.normal(subkey, shape=states.shape)
    observations = states + noise
    observation_times = observation_times[:30]
    observations = observations[:30,:]
    subsample = 6

Next, define our likelihood function and prior.

.. code-block:: python

    def observation_log_likelihood(observation, state):
        partition = jnp.power(2 * pi, -observations.shape[1]/2.0)
        return jnp.log(partition) -1 * jnp.sum(jnp.square(observation - state))/(2*noise_std**2)

    def state_log_prior(state):
        return -1 * jnp.log(9)