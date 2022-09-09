<img
  src="sumo.jpeg"
  alt="A sumo"
  width=150>

## Sumomo: Surrogate Modelling and Optimisation
Sumomo is a Python package for surrogate modelling formulating optimisation problems using *Pyomo*. Surrogate models include Gaussian processes and neural networks for regression and classification. sumomo enables an entire surrogate modelling workflow and abstracted *Pyomo* formulations to be incorporated into larger optimisation problems.

## Example
```python
from scripts.functions import BlackBox
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    mean_squared_error
)

from sumomo import DataHandler, GPR, GPC


bb = BlackBox()
n_samples = 100
space = [(-3.0, 3.0), (-3.0, 3.0)]

data = DataHandler()
data.init(n_samples, space)
data.y = bb.sample_y(data.x)
data.t = bb.sample_t(data.x)
data.split(test_size=0.2)
data.scale()

regressor = GPR()
classifier = GPC()

regressor.fit(data.x_train_, data.y_train_)
classifier.fit(data.x_train_, data.t_train)

predictions = regressor.predict(data.x_test_)
predictions = data.inv_scale_y(predictions)

probabilities, classes = classifier.predict(
    data.x_test_, return_class=True)

error = mean_squared_error(data.y_test, predictions)
precision = precision_score(data.t_test, classes)
recall = recall_score(data.t_test, classes)

print(error)
print(precision)
print(recall)
```
