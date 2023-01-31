
## OODX: Object-Orientated Derivative-Free Optimisation
OODX is a Python package, designed to be used in conjunction with *Pyomo*, for formulating surrogate models within larger decision-making problems. Surrogate models include Gaussian processes and neural networks for both regression and classification. OODX contains the following modelling objects:

* `DataBlock` for generating initial sampling strategies as well as processing and storing data
* `GPR` and `GPC` for Gaussian process regression and classification models, respectively
* `NN` for neural networks for regression or classification
* `MPBlock` for building abstracted mathematical programming formulations from trained surrogate models
* `AdaptiveSampler` for generating adaptive samples for surrogate modelling
