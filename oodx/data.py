import numpy as np
import pandas as pd
import math
from skopt.space import Space
from skopt.sampler import Lhs, Sobol


class DataBlock:
    def __init__(self, inputs: list = [], outputs: list = [], targets: bool = False):
        """
        inputs         -       list of strings for input variables names
        outputs        -       list of strings for input variables names
        targets        -       Boolean variable True if classification targets to be included

        Returns a DataBlock instance with attributes:
        data           -       empty data frame with inputs, outputs, targets headers
        """
        headers = inputs + outputs + targets * ["t"]
        self.data = pd.DataFrame(columns=headers)
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets

    @property
    def mean_inputs(self):
        """
        Returns a Series of input means
        """
        return self.data[self.inputs].mean()

    @property
    def std_inputs(self):
        """
        Returns a Series of input standard deviations
        """
        return self.data[self.inputs].std(ddof=0)

    @property
    def scaled_inputs(self):
        """
        Returns a DataFrame of normalised input variables
        """
        return (self.data[self.inputs] - self.mean_inputs) / self.std_inputs

    @property
    def mean_outputs(self):
        """
        Returns a Series of output means accounting for infeasibilities
        """
        if self.targets:
            return self.data[self.data.t == 1][self.outputs].mean()
        else:
            return self.data[self.outputs].mean()

    @property
    def std_outputs(self):
        """
        Returns a Series of output standard deviations accounting for infeasibilities
        """
        if self.targets:
            return self.data[self.data.t == 1][self.outputs].std(ddof=0)
        else:
            return self.data[self.outputs].std(ddof=0)

    @property
    def scaled_outputs(self):
        """
        Returns a DataFrame of normalised output variables accounting for infeasibilities
        """
        return (self.data[self.outputs] - self.mean_outputs) / self.std_outputs

    @property
    def scaled_space(self):
        """
        Returns a list of scaled lower, upper bound tuples in each input dimension
        """
        space = []
        for i, val in enumerate(self.space):
            lb = (val[0] - self.mean_inputs[i]) / self.std_inputs[i]
            ub = (val[1] - self.mean_inputs[i]) / self.std_inputs[i]
            space.append([lb, ub])
        return space

    def static_sample(self, n: int, space: list, method: str = "lhs"):
        """
        n              -       number of input samples to generate
        space          -       list of lower, upper bound tuples in each input dimension
        method         -       sampling method as string
        Returns a DataBlock instance with static input samples
        """
        self.space = space
        m = len(space)

        if method == "lhs":
            lhs = Lhs(criterion="maximin", iterations=1000)
            input_space = Space(space)
            self.data[self.inputs] = lhs.generate(input_space.dimensions, n)
        elif method == "random":
            lbs, ubs = [val[0] for val in space], [val[1] for val in space]
            self.data[self.inputs] = np.random.uniform(lbs, ubs, (n, m))
        elif method == "sobol":
            sobol = Sobol()
            input_space = Space(space)
            self.data[self.inputs] = sobol.generate(input_space.dimensions, n)
        elif method == "grid":
            n_root = math.ceil(n ** (1 / m))
            x_coords = [np.linspace(*space[i], n_root).tolist() for i in range(m)]
            mg = np.meshgrid(*x_coords)
            grid = mg[0].reshape(-1, 1)
            for i in range(1, len(mg)):
                grid = np.c_[grid, mg[i].reshape(-1, 1)]
            np.random.shuffle(grid)
            self.data[self.inputs] = grid[:n]

    def scale_inputs(self, x):
        """
        x              -       DataFrame of input variables to be scaled
        Returns a DataFrame of normalised x
        """
        return (x - self.mean_inputs.values) / self.std_inputs.values

    def inv_scale_inputs(self, x):
        """
        x              -       DataFrame of input variables to be inverse scaled
        Returns a DataFrame of denormalised x
        """
        return x * self.std_inputs.to_numpy() + self.mean_inputs.to_numpy()

    def scale_outputs(self, y):
        """
        y              -       DataFrame of output variables to be scaled
        Returns a DataFrame of normalised y
        """
        return (y - self.mean_outputs) / self.std_outputs

    def inv_scale_outputs(self, y):
        """
        y              -       DataFrame of output variables to be inverse scaled
        Returns a DataFrame of denormalised y
        """
        return y * self.std_outputs.values + self.mean_outputs.values

    def scale_space(self, space):
        return [
            [
                (bounds[0] - self.mean_inputs.values[i]) / self.std_inputs.values[i],
                (bounds[1] - self.mean_inputs.values[i]) / self.std_inputs.values[i],
            ]
            for i, bounds in enumerate(space)
        ]
