<img
  src="sumo.jpeg"
  alt="A sumo"
  width=150>

## Sumomo: Surrogate Modelling and Optimisation
Sumomo is a Python package for surrogate modelling formulating optimisation problems using *Pyomo*. Surrogate models include Gaussian processes and neural networks for regression and classification. sumomo enables an entire surrogate modelling workflow and abstracted *Pyomo* formulations to be incorporated into larger optimisation problems.

## Example
```python
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    mean_squared_error
)

from scripts.functions import BlackBox
from sumomo import DataHandler, GPR, GPC


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
