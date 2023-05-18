===
API 
===

Manifold 
--------

The core functionality is found in the ``trajectory_manifold.manifold`` module.

.. automodule:: trajectory_manifold.manifold
   :members: system_sensitivity, system_pushforward_weight, system_pushforward_weight_reweighted

   .. py:class:: trajectory_manifold.manifold.SolverParameters(NamedTuple)

      Stores Information for ODE Solvers. 

      Records the parameters for solving an ODE using Diffrax,
      including the solver, tolerances, output grid size, and time horizon
    
      :param relative_tolerance: Relative tolerance for the ODE solution
      :param absolute_tolerance: Absolute tolerance for the ODE solution
      :param step_size: Output mesh size. Note: Does not impact internal computations.
      :param time_horizon: Length of the solution in seconds.
      :param solver: The particular ODE solver to use.


Estimation 
----------

The module ``trajectory_manifold.estimation`` contains functions related to probability.

.. automodule:: trajectory_manifold.estimation 
   :members: 

Optimization 
------------

The module ``trajectory_manifold.optimize`` contains tools for optimization on the
manifold of feasible trajectories.
The main approach is the computation of pullbacks of gradients into the state space.

.. automodule:: trajectory_manifold.optimize
   :members:

Helpers
-------

The module ``trajectory_manifold.helpers`` contains a collection of helper functions
for the small modifications to linear algebra operations required in the
project.

.. automodule:: trajectory_manifold.helpers 
   :members:

Examples 
--------

The module ``trajectory_manifold.examples`` contains a collection of example systems 
to be used with the trajectory forecasting work.
It contains functions which generate vector fields for a linear system,
a periodic system, and a chaotic system.

.. automodule:: trajectory_manifold.examples
   :members:
