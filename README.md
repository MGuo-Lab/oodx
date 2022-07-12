<img
  src="sumo.jpeg"
  alt="A sumo"
  width=150>

# SUMOMO: Surrogate modelling and optimisation modelling
SUMOMO is a Python package for surrogate modelling and optimisation modelling.

## Neural networks
Neural networks are implemented using *PyTorch*.

## Gaussian processes
Gaussian process regression models are implemented using `GaussianProcessRegressor` from *scikit-learn* whilst Gaussian process classification is implemented using *NumPy*.

## Pyomo formulations
Each surrogate model formulations is coded as a *Pyomo* Block.

## Application programming interface
The surrogate modelling with neural networks and Gaussian processes, as well as the *Pyomo* formulations are made available in a single `api` object.

## Example
```python
import pyomo.environ as pyo
from sumomo import api
```
