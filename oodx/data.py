import numpy as np
import pandas as pd
import math
from skopt.space import Space
from skopt.sampler import Lhs, Sobol
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataBlock:
    def __init__(self, inputs:list, outputs:list, targets:bool=False):
        '''
        inputs         -       list of strings for input variables names
        outputs        -       list of strings for input variables names
        targets        -       Boolean variable True if classification targets to be included
        
        Returns a DataBlock instance with attributes:
        data           -       empty data frame with inputs, outputs, targets headers
        '''
        headers = inputs + outputs + targets * ['t']
        self.data = pd.DataFrame(columns=headers)
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
    
    @property
    def mean_inputs(self):
        '''
        Returns a Series of input means
        '''
        return self.data[self.inputs].mean()
    
    @property
    def std_inputs(self):
        '''
        Returns a Series of input standard deviations
        '''
        return self.data[self.inputs].std()

    @property
    def scaled_inputs(self):
        '''
        Returns a DataFrame of normalised input variables
        '''
        return (self.data[self.inputs] - self.mean_inputs) / self.std_inputs
    
    @property
    def mean_outputs(self):
        '''
        Returns a Series of output means accounting for infeasibilities
        '''
        if self.targets:
            return self.data[self.data.t==1][self.outputs].mean()
        else:
            return self.data[self.outputs].mean()
    
    @property
    def std_outputs(self):
        '''
        Returns a Series of output standard deviations accounting for infeasibilities
        '''
        if self.targets:
            return self.data[self.data.t==1][self.outputs].std()
        else:
            return self.data[self.outputs].std()
    
    @property
    def scaled_outputs(self):
        '''
        Returns a DataFrame of normalised output variables accounting for infeasibilities
        '''
        return (self.data[self.outputs] - self.mean_outputs) / self.std_outputs
    
    @property
    def scaled_space(self):
        '''
        Returns a list of scaled lower, upper bound tuples in each input dimension
        '''
        space = []
        for i, val in enumerate(self.space):
            lb = (val[0] - self.mean_inputs[i]) / self.std_inputs[i]
            ub = (val[1] - self.mean_inputs[i]) / self.std_inputs[i]
            space.append( [lb, ub] )
        return space

    def static_sample(self, n:int, space:list, method:str='lhs'):
        '''
        n              -       number of input samples to generate
        space          -       list of lower, upper bound tuples in each input dimension
        method         -       sampling method as string
        Returns a DataBlock instance with static input samples
        '''
        self.space = space
        m = len(space)

        if method == 'lhs':
            lhs = Lhs(criterion='maximin', iterations=1000)
            input_space = Space(space)
            self.data[self.inputs] = lhs.generate(input_space.dimensions, n)
        elif method == 'random':
            lbs, ubs = [val[0] for val in space], [val[1] for val in space]
            self.data[self.inputs] = np.random.uniform(lbs, ubs, (n, m))
        elif method == 'sobol':
            sobol = Sobol()
            input_space = Space(space)
            self.data[self.inputs] = sobol.generate(input_space.dimensions, n)
        elif method == 'grid':
            n_root = math.ceil(n ** (1/m))
            x_coords = [np.linspace(*space[i], n_root) for i in range(m)]
            mg = np.meshgrid(*x_coords)
            grid = np.array([val.ravel() for val in mg]).reshape((n_root**m, m))
            np.random.shuffle(grid)
            self.data[self.inputs] = grid[:n]
    
    def scale_inputs(self, x):
        '''
        x              -       DataFrame of input variables to be scaled
        Returns a DataFrame of normalised x
        '''
        return (x - self.mean_inputs.values) / self.std_inputs.values
    
    def inv_scale_inputs(self, x):
        '''
        x              -       DataFrame of input variables to be inverse scaled
        Returns a DataFrame of denormalised x
        '''
        return x * self.std_inputs.to_numpy() + self.mean_inputs.to_numpy()
    
    def scale_outputs(self, y):
        '''
        y              -       DataFrame of output variables to be scaled
        Returns a DataFrame of normalised y
        '''
        return (y - self.mean_inputs) / self.std_inputs
    
    def inv_scale_outputs(self, y):
        '''
        y              -       DataFrame of output variables to be inverse scaled
        Returns a DataFrame of denormalised y
        '''
        return y * self.std_outputs.values + self.mean_outputs.values
    
    def scale_space(self, space):
        return [[
            (bounds[0] - self.mean_inputs.values[i]) / self.std_inputs.values[i], 
            (bounds[1] - self.mean_inputs.values[i]) / self.std_inputs.values[i]
            ] for i, bounds in enumerate(space)]


class DataHandler:
    def __init__(self):
        # input sample space
        self.space = None
        # input, output, and training variables
        self.x = None
        self.y = None
        self.t = None
        # training and testing variables
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.t_train = None
        self.t_test = None
        # scaled variables
        self.space_ = None
        self.x_ = None
        self.y_ = None
        self.x_train_ = None
        self.x_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        # scaling paramters / statistical moments
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.x_train_mean = None
        self.x_train_std = None
        self.y_train_mean = None
        self.y_train_std = None
    
    def init(self, n_samples, space, n_outputs=1, method='lhs'):
        '''
        n_samples         -       number of inputs samples
        space             -       input space
        n_outputs         -       number of ouput dimensions to initialise
        method            -       sampling method: random, lhs, sobol, grid
        '''

        # save space and initialise outputs and targets
        self.space = space
        self.y = np.zeros((n_samples, n_outputs))
        self.t = np.ones((n_samples, 1))

        if method == 'random':
            mat = np.random.rand(n_samples, len(self.space))
            samples = np.zeros_like(mat)
            for i in range(n_samples):
                for j in range(len(self.space)):
                    samples[i][j] = mat[i][j] * (self.space[j][1] - self.space[j][0]) + self.space[j][0]
            self.x = samples

        elif method == 'lhs':
            lhs = Lhs(criterion='maximin', iterations=1000)
            input_space = Space(self.space)
            lhs_samples = lhs.generate(input_space.dimensions, n_samples)
            self.x = np.array(lhs_samples)

        elif method == 'sobol':
            sobol = Sobol()
            input_space = Space(self.space)
            sobol_samples = sobol.generate(input_space.dimensions, n_samples)
            self.x = np.array(sobol_samples)
        
        elif method == 'grid':
            m = len(space)
            n = math.ceil(n_samples ** (1/m))
            x1, x2 = np.linspace(*self.space[0], n), np.linspace(*self.space[1], n)
            x1_grid, x2_grid = np.meshgrid(x1, x2)
            grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
            samples = np.array(grid)
            np.random.shuffle(samples)
            self.x = samples[:n_samples]
    

    def split(self, test_size=0.3):
        # train-test split on x, y, t
        self.x_train, self.x_test, self.y_train, self.y_test, self.t_train, self.t_test = train_test_split(
            self.x, self.y, self.t, test_size=test_size)
    
    
    def scale(self):
        # normalise x
        scaler = StandardScaler()
        self.x_ = scaler.fit_transform(self.x)
        self.x_mean, self.x_std = scaler.mean_, scaler.scale_
        
        # normalise y only on converged data
        y_con = self.y[self.t.ravel() == 1, :]
        scaler.fit(y_con)
        self.y_mean, self.y_std = scaler.mean_, scaler.scale_
        self.y_ = (self.y - self.y_mean) / self.y_std
        
        if self.x_train is not None:
            # normalise x_train
            self.x_train_ = scaler.fit_transform(self.x_train)
            self.x_train_mean, self.x_train_std = scaler.mean_, scaler.scale_
            # normalise x_test using training moments
            self.x_test_ = (self.x_test - self.x_train_mean) / self.x_train_std
            # normalise y_train only on converged data
            y_train_con = self.y_train[self.t_train.ravel() == 1, :]
            scaler.fit(y_train_con)
            self.y_train_mean, self.y_train_std = scaler.mean_, scaler.scale_
            self.y_train_ = (self.y_train - self.y_train_mean) / self.y_train_std
            # normalise y_test using training moments
            self.y_test_ = (self.y_test - self.y_train_mean) / self.y_train_std
            # normalise space using training moments
            self.space_ = [] 
            for i, val in enumerate(self.space):
                lb = (val[0] - self.x_train_mean[i]) / self.x_train_std[i]
                ub = (val[1] - self.x_train_mean[i]) / self.x_train_std[i]
                self.space_.append( (lb, ub) )
        else:
            # normalise space using x moments
            self.space_ = [] 
            for i, val in enumerate(self.space):
                lb = (val[0] - self.x_mean[i]) / self.x_std[i]
                ub = (val[1] - self.x_mean[i]) / self.x_std[i]
                self.space_.append( (lb, ub) )
            
    
    def scale_space(self, space):
        # normalise space using x moments
        new_space = []
        for i, val in enumerate(space):
            if self.x_train is not None:
                lb = (val[0] - self.x_train_mean[i]) / self.x_train_std[i]
                ub = (val[1] - self.x_train_mean[i]) / self.x_train_std[i]
            else:
                lb = (val[0] - self.x_mean[i]) / self.x_std[i]
                ub = (val[1] - self.x_mean[i]) / self.x_std[i]
            new_space.append( [lb, ub] )
        return new_space


    def inv_scale_x(self, x):
        output = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.x_train is not None:
                    output[i, j] = x[i, j] * self.x_train_std[j] + self.x_train_mean[j]
                else:
                    output[i, j] = x[i, j] * self.x_std[j] + self.x_mean[j]
        return output
    

    def scale_x(self, x):
        output = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.x_train is not None:
                    output[i, j] = (x[i, j] - self.x_train_mean[j]) / self.x_train_std[j]
                else:
                    output[i, j] = (x[i, j] - self.x_mean[j]) / self.x_std[j]
        return output
    

    def inv_scale_y(self, y):
        output = np.zeros_like(y)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if self.y_train is not None:
                    output[i, j] = y[i, j] * self.y_train_std[j] + self.y_train_mean[j]
                else:
                    output[i, j] = y[i, j] * self.y_std[j] + self.y_mean[j]
        return output
    
    
    def scale_y(self, y):
        output = np.zeros_like(y)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if self.x_train is not None:
                    output[i, j] = (y[i, j] - self.y_train_mean[j]) / self.y_train_std[j]
                else:
                    output[i, j] = (y[i, j] - self.y_mean[j]) / self.y_std[j]
        return output
        