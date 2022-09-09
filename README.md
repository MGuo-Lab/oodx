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
