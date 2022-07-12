<img
  src="sumo.jpeg"
  alt="A sumo"
  width=100>

# SUMOMO
This package enables surrogate modelling and *Pyomo* optimisaton formulations for black box models.

## Neural networks
Neural networks are implemented using *PyTorch*.

## Gaussian processes
Gaussian process regression models are implemented using `GaussianProcessRegressor` from *scikit-learn* whilst Gaussian process classification is implemented using *NumPy*.

## Pyomo formulations
Each surrogate model formulations is coded as a *Pyomo* Block.

## Application programming interface
The surrogate modelling with neural networks and Gaussian processes, as well as the *Pyomo* formulations are made available in a single `api` object.
