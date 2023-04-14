=====
Usage
=====

Installation
------------

Installation is to come later. This code will eventually be turned into a 
package to be installed with pip.

Quick Start 
-----------

Trajectory estimation over any finite time-horizon is a reparameterization
of the state estimation problem for a system.
Thus, the only change to classical estimation methods is to incorporate the
reparameterization to the objective function.

The key function in the library is ``manifold.system_pushforward_weight``.


Given a description of the dynamics and forecasting objective, this function
constructs the associated reweighting for a given point.
