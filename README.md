<img
  src="sumo.jpeg"
  alt="A sumo"
  width=150>

## Sumomo: Surrogate Modelling and Optimisation
Sumomo is a Python package, designed to be used in conjunction with *Pyomo*, for formulating surrogate models within larger decision-making problems. Surrogate models include Gaussian processes and neural networks for both regression and classification. Sumomo contains objects:

* `DataHandler` for generating initial sampling strategies as well as processing and storing data
* `GPR` and `GPC` for Gaussian process regression and classification models, respectively
* `NN` for neural networks for regression or classification
* `AdaptiveSampler` for generating adaptive samples for surrogate modelling
* `BlockFormulation` for building abstracted *Pyomo* formulations from trained surrogate models

## Surrogate modelling example
```python
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    mean_squared_error
)
from sumomo import DataHandler, GPR, GPC
from sumomo.examples import BlackBox


# initialise data handler
data = DataHandler()
n_samples = 100
space = [(-3.0, 3.0), (-3.0, 3.0)]
data.init(n_samples, space)

# sample data from the black box
bb = BlackBox()
data.y = bb.sample_y(data.x)
data.t = bb.sample_t(data.x)

# train-test-split and standardise data
data.split(test_size=0.2)
data.scale()

# initilise and fit regression model
regressor = GPR()
regressor.fit(data.x_train_, data.y_train_)

# initialise and fit classification model
classifier = GPC()
classifier.fit(data.x_train_, data.t_train)

# make regression and classification predictions
predictions = regressor.predict(data.x_test_)
predictions = data.inv_scale_y(predictions)
probabilities, classes = classifier.predict(
    data.x_test_, return_class=True)

# evaluate validation metrics
error = mean_squared_error(data.y_test, predictions)
precision = precision_score(data.t_test, classes)
recall = recall_score(data.t_test, classes)
print(error)
print(precision)
print(recall)

```

## Surrogate-based optimisation example

```python
import pyomo.environ as pyo
from sumomo.examples import BlackBox
from sumomo import (
    DataHandler, GPR, GPC, BlockFormulation
)


# initialise data handler
data = DataHandler()
n_samples = 100
space = [(-3.0, 3.0), (-3.0, 3.0)]
data.init(n_samples, space)

# sample data from the black box
bb = BlackBox()
data.y = bb.sample_y(data.x)
data.t = bb.sample_t(data.x)

# initilise and fit regression model
regressor = GPR()
regressor.fit(data.x, data.y)

# initialise and fit classification model
classifier = GPC()
classifier.fit(data.x, data.t)

# pyomo formulation, sets, and variables
omo = pyo.ConcreteModel()
omo.n_inputs = set(range(len(data.space)))
omo.inputs = pyo.Var(omo.n_inputs, bounds=data.space)
omo.output = pyo.Var()
omo.proba = pyo.Var()

# feasibility constraint 
omo.feasibility_con = pyo.Constraint(expr=
    omo.proba >= 0.5 
)

# objective function to maximise performance
omo.obj = pyo.Objective(
    expr=omo.output, sense=pyo.maximize)

# formulate pyomo blocks for regressor and classifier
omo.reg = pyo.Block(rule= 
    BlockFormulation(regressor).rule()
)
omo.cla = pyo.Block(rule=
    BlockFormulation(classifier).rule()
)

# connect pyomo model input and output to the surrogate models
omo.c = pyo.ConstraintList()
omo.c.add( omo.output == omo.reg.outputs[0] )
omo.c.add( omo.proba == omo.cla.outputs[0] )
for i in omo.n_inputs:
    omo.c.add( omo.inputs[i] == omo.reg.inputs[i] )
    omo.c.add( omo.inputs[i] == omo.cla.inputs[i] )

# solve
solver = pyo.SolverFactory('baron')
solver.solve(omo)

```

