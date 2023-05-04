
## OODX: Object-Orientated Derivative-Free Optimisation
OODX is a Python package, designed to be used in conjunction with Pyomo, for formulating surrogate models within larger decision-making problems. Surrogate models include Gaussian processes and neural networks for both regression and classification. OODX contains the following modelling objects:

* `DataBlock` for generating initial sampling strategies as well as processing and storing data
* `GPR` and `GPC` for Gaussian process regression and classification models, respectively
* `NN` for neural networks for regression or classification
* `MPBlock` for building abstracted mathematical programming formulations from trained surrogate models
* `AdaptiveSampler` for generating adaptive samples for surrogate modelling

# Installation
```
pip install oodx
```

# Example
```python
from oodx import DataBlock, GPR, MPBlock
import pyomo.environ as pyo
from utils import func_1d as bb


# sample data from black box
db = DataBlock(inputs=['x'], outputs=['y'])
db.static_sample(n=6, space=[[-5.0, 5.0]])
db.data.y = bb(db.data.x)
print(db.data)

# fit a GP model
gp = GPR()
gp.fit(db.data.x.values.reshape(-1, 1), db.data.y.values.reshape(-1, 1))

# formulate optimisation problem embedding GP
m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=db.space[0])
m.gp_block = MPBlock(gp).get_formulation()
m.c = pyo.Constraint(expr= m.x == m.gp_block.inputs[0])
m.obj = pyo.Objective(expr=m.gp_block.outputs[0], sense=pyo.minimize)

# setup optimisation solver and solve
solver = pyo.SolverFactory('baron')
solver.solve(m)
```

![example](https://user-images.githubusercontent.com/45121699/236180495-484838e6-0364-47cc-bef2-9ced10b88716.png)

