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

Trajectory estimation over any finite time-horizon is a reparameterization
of the state estimation problem for a system.
Thus, the only change to classical estimation methods is to incorporate the
reparameterization to the objective function.

The key function in the library is ``manifold.system_pushforward_weight``.


Given a description of the dynamics and forecasting objective, this function
constructs the associated reweighting for a given point.
