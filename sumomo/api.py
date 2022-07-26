import numpy as np
from numpy.linalg import det
from skopt.space import Space
from skopt.sampler import Lhs, Sobol
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyomo.environ as pyo
from scipy.spatial import Delaunay
import itertools
import math

from .gp import GPR, GPC
from .nn import NN
from .formulations import BlockFormulation


class API:
    def __init__(self, n_samples, space, n_outputs=1, method='lhs'):
        # input sample space
        self.space = space
        # input, output, and training variables
        self.x = self._get_inputs(n_samples, method)
        self.y = np.zeros((n_samples, n_outputs))
        self.t = np.ones((n_samples, 1))
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
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.x_train_mean = None
        self.x_train_std = None
        self.y_train_mean = None
        self.y_train_std = None
        # model objects
        self.regressor = None
        self.classifier = None
        # adaptive samples
        self.delaunay = None

    def _get_inputs(self, n_samples, method='lhs'):
        '''
        n_samples         -       number of inputs samples
        method            -       sampling method: random, lhs, sobol, grid
        '''
        
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
            n = int(np.sqrt(n_samples))
            x1, x2 = np.linspace(*self.space[0], n), np.linspace(*self.space[1], n)
            x1_grid, x2_grid = np.meshgrid(x1, x2)
            grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
            self.x = np.array(grid)
        
        return self.x
    
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
    
    def init_regressor(self, type='gp', **kwargs):
        if type == 'gp':
            noise = 0.0
            if 'noise' in kwargs:
                noise = kwargs['noise']
            self.regressor = GPR(noise)
        elif type == 'nn':
            activation = 'tanh'
            if 'hidden_layers' in kwargs:
                hidden_layers = kwargs['hidden_layers']
                layers = [self.x.shape[1], *hidden_layers, self.y.shape[1]]
            if 'activation' in kwargs:
                activation = kwargs['activation']
            self.regressor = NN(layers=layers, activation=activation)
        # print(self.regressor)
    
    def init_classifier(self, type='gp', **kwargs):
        if type == 'gp':
            self.classifier = GPC()
        elif type == 'nn':
            activation = 'sigmoid'
            if 'hidden_layers' in kwargs:
                hidden_layers = kwargs['hidden_layers']
                layers = [len(self.x.T), *hidden_layers, len(self.y.T)]
            if 'activation' in kwargs:
                activation = kwargs['activation']
            self.classifier = NN(layers=layers, activation=activation)
        # print(self.classifier)
        
    def test(self, method='mse'):
        if self.x_test is None:
            pred = self.regressor.predict(self.x_)
            true = self.y_
        else:
            pred = self.regressor.predict(self.x_test_)
            true = self.y_test_
        
        if method == 'mse':
            sq_err = (pred - true) ** 2
            mse = np.mean(sq_err)
            return mse
        
        if method == 'rmse':
            return np.sqrt(self.test('mse'))
    
    def formulate(self, model):
        return pyo.Block(rule=BlockFormulation(model, self.space_).rule())
    
    def check_convergence(self):
        # TODO
        pass

    def delaunay_explore(self):
        centroids, sizes = self._get_delaunay_centroids_and_sizes()
        centroid = centroids[sizes.index(max(sizes))]
        return centroid
    
    def delaunay_exploit(self, sense):
        centroids, sizes = self._get_delaunay_centroids_and_sizes()

        if sense == 'max':
            if self.y_train is None:
                index = list(self.y).index(max(list(self.y)))
            else:
                index = list(self.y_train).index(max(list(self.y_train)))
        elif sense == 'min':
            if self.y_train is None:
                index = list(self.y).index(min(list(self.y)))
            else:
                index = list(self.y_train).index(min(list(self.y_train)))

        exploit_areas = [0] * len(centroids)
        ind = np.where(np.any(self.delaunay.simplices == index, axis = 1))[0]
        for i in ind:
            exploit_areas[i] = sizes[i]
        centroid = centroids[exploit_areas.index(max(exploit_areas))]
        
        return centroid
    
    def _get_delaunay_centroids_and_sizes(self):
        vertices = np.array(list(itertools.product(*self.space)))
        if self.x_train is None:
            points = np.r_[self.x, vertices]
        else:
            points = np.r_[self.x_train, vertices]
        points = self.x   # TODO - sample at vertices to get tightest convex hull on sample space then adaptive sample?
        self.delaunay = Delaunay(points)
        
        centroids = np.zeros((self.delaunay.simplices.shape[0], self.x.shape[1]))
        for i, s in enumerate(self.delaunay.simplices):
            vals = points[s, :]
            centroids[i, :] = [sum(vals[:, j]) / vals.shape[0] for j in range(vals.shape[1])]
        
        sizes = [0] * len(self.delaunay.simplices)
        for i, s in enumerate(self.delaunay.simplices):
            dist = np.delete(points[s] - points[s][-1], -1, 0)
            sizes[i] = abs(1 / math.factorial(points.shape[1]) * det(dist))
        
        return centroids, sizes

    def max_gp_uncertainty(self):
        # TODO
        pass

    def _ei(self, x_new, xi=0.0):
        mu, sigma = self.regressor.predict(x_new, return_std=True)
        sigma = sigma.reshape(-1, 1)

        if self.y_train is None:
            y_max = np.max(self.y_)
        else:
            y_max = np.max(self.y_train_)

        with np.errstate(divide='warn'):
            imp = mu - y_max - xi
            Z = imp / sigma
            from scipy.stats import norm
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
    def _modified_ei(self, x_new):
        mu, sigma = self.regressor.predict(x_new, return_std=True)
        sigma = sigma.reshape(-1, 1)

        if self.y_train is None:
            y_max = np.max(self.y_)
        else:
            y_max = np.max(self.y_train_)

        with np.errstate(divide='warn'):
            imp = mu - y_max
            Z = imp / sigma
            from scipy.stats import norm
            ei = sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def max_expected_improvement(self, aquisition='ei'):
        min_val = 1
        def min_ei(x):
            return -self._ei(x.reshape(-1, self.x.shape[1]))
        def min_modified_ei(x):
            return -self._modified_ei(x.reshape(-1, self.x.shape[1]))
        
        lb = [val[0] for val in self.space_]
        ub = [val[1] for val in self.space_]

        for x0 in np.random.uniform(lb, ub, size=(50, self.x.shape[1])):
            from scipy.optimize import minimize
            if aquisition == 'modified_ei':
                res = minimize(min_modified_ei, x0=x0, bounds=self.space_, method='L-BFGS-B')
            elif aquisition == 'ei':
                res = minimize(min_ei, x0=x0, bounds=self.space_, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x               
        return min_x.reshape(-1, self.x.shape[1])
