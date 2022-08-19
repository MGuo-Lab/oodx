import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs, Sobol
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        
        if self.y_train is not None:
            # normalise y_train only on converged data
            y_train_con = self.y_train[self.t_train.ravel() == 1, :]
            scaler.fit(y_train_con)
            self.y_train_mean, self.y_train_std = scaler.mean_, scaler.scale_
            self.y_train_ = (self.y_train - self.y_train_mean) / self.y_train_std
            # normalise y_test using training moments
            self.y_test_ = (self.y_test - self.y_train_mean) / self.y_train_std
    

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
        