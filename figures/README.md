# Figure Generation

In order to enable faster experimentation, the simulations in this work are split into files which generate intermediate results.

The figures were generated through grid-based approximations.

Figure 1 and Figure 2 can be generated through the file `lotka-weights.py` followed by `lotka-estimate.py`.
- `lotka-weights.py` constructs the pushforward weights for a 10 second time horizon.
- `lotka-estimate.py` reads in these weights and uses them for the estimation and plotting.

Figure 3 can be generated through running `lotka-sensitivity.py`, then `lotka-quant.py`, then `lotka-quant-plot.py`.
- `lotka-sensitivity.py` constructs the pushforward weights for various time horizons
- `lotka-quant.py` runs the simulations
- `lotka-quant-plot.py` plots the results.


NOTE:
    Figure Generation is currently broken in the main branch.
    To reproduce the figures from the paper, install the v0.0.1 release.