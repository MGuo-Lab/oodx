import numpy as np
from numpy.linalg import det
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import pyomo.environ as pyo
import itertools
import math

from .formulations import BlockFormulation


class AdaptiveSampler:
    def __init__(self, space, model=None, classifier=None):
        self.space = space
        self.model = model
        self.classifier = classifier

    def max_gp_std(self, solver=None):
        if self.classifier is not None:
            return self._constrained_max_gp_std(solver)
        else:
            return self._scipy_max_gp_std()
    
    def max_dt(self, x, solver=None):
        # get centroids and sizes
        centroids, sizes = self._get_delaunay_centroids_and_sizes(x)
        # init pyomo model
        m = pyo.ConcreteModel()
        # dt declarations
        m.n_dt = set(range(len(centroids)))
        m.y = pyo.Var(m.n_dt, domain=pyo.Binary)
        m.max_dt = pyo.Var()
        # input declarations
        m.n_inputs = set(range(len(self.space)))
        m.inputs = pyo.Var(m.n_inputs, bounds=self.space)
        # constraints
        m.exactly_one_con = pyo.Constraint(expr= sum(m.y[i] for i in m.n_dt) == 1 )
        m.feas = pyo.Block(rule=BlockFormulation(self.classifier).rule())
        m.feasibility_con = pyo.Constraint(expr= m.feas.outputs[0] >= 0.5 )
        m.c = pyo.ConstraintList()
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == sum(centroids[k, i] * m.y[k] for k in m.n_dt) )
            m.c.add( m.inputs[i] == m.feas.inputs[i] )
        # objective
        m.obj = pyo.Objective(expr= sum(m.y[i] * sizes[i] for i in m.n_dt), sense=pyo.maximize)
        # solve and return solution
        res = solver.solve(m, tee=True)
        new = np.fromiter(m.inputs.extract_values().values(), dtype=float).reshape(1, -1)
        return new

    def _scipy_max_gp_std(self):
        def func(x):
            return - self.model.predict(x.reshape(1, -1), return_std=True)[1]
       
        min_val = 1
        lb = [val[0] for val in self.space]
        ub = [val[1] for val in self.space]
        for x0 in np.random.uniform(lb, ub, size=(50, len(self.space))):
            res = minimize(func, x0=x0, bounds=self.space, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
        return min_x.reshape(1, -1)

    def _constrained_max_gp_std(self, solver):
        # initialise pyomo model
        m = pyo.ConcreteModel()
        # formulate pyomo block for gpr std and gpc
        m.mdl = pyo.Block(rule=BlockFormulation(self.model).rule(return_std=True))
        m.feas = pyo.Block(rule=BlockFormulation(self.classifier).rule())
        
        m.n_inputs = set(range(len(self.space)))
        m.inputs = pyo.Var(m.n_inputs, bounds=self.space)
        m.feasibility_con = pyo.Constraint(expr= m.feas.outputs[0] >= 0.5 )
        m.obj = pyo.Objective(expr=m.mdl.outputs[0], sense=pyo.minimize)
        
        # connect pyomo model input and output to the surrogate models
        m.c = pyo.ConstraintList()
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.mdl.inputs[i] )
            m.c.add( m.inputs[i] == m.feas.inputs[i] )
        
        res = solver.solve(m, tee=True)
        x = np.fromiter(m.inputs.extract_values().values(), dtype=float).reshape(1, -1)

        return x
    
    def _get_delaunay_centroids_and_sizes(self, x):
        self.delaunay = Delaunay(x)
        
        centroids = np.zeros((self.delaunay.simplices.shape[0], x.shape[1]))
        for i, s in enumerate(self.delaunay.simplices):
            vals = x[s, :]
            centroids[i, :] = [sum(vals[:, j]) / vals.shape[0] for j in range(vals.shape[1])]
        
        sizes = [0] * len(self.delaunay.simplices)
        for i, s in enumerate(self.delaunay.simplices):
            dist = np.delete(x[s] - x[s][-1], -1, 0)
            sizes[i] = abs(1 / math.factorial(x.shape[1]) * det(dist))
        
        return centroids, sizes
