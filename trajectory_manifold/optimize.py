"""This module is for MCMC optimization methods.

The main approach in this module is to use simulated annealing.
This is due to the dependence of the first-order derivative of the
posterior distribution of trajectories on the second-order sensitivity
of the solution to the initial conditions.
"""
from typing import Callable

def simulated_annealing(
    f: Callable,
    proposal_variable: Callable, # Random variable
    sample_count: int,
):
    """Optimize a given function through simulated annealing."""
    pass 

