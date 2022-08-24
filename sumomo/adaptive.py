import numpy as np
from numpy.linalg import det
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.stats import norm
import pyomo.environ as pyo
import math

from .formulations import BlockFormulation


class AdaptiveSampler:
    def __init__(self, space, model=None, classifier=None):
        self.space = space
        self.model = model
        self.classifier = classifier
        self.delaunay = None
    
    def expected_improvement(self, y, sense='max', aquisition='ei', xi=0.0, solver=None):
        if self.classifier is not None:
            return self._constrained_modified_ei(y, sense, solver)
        else:
            def min_ei(x):
                return -self._ei(x.reshape(1, -1), y=y, sense=sense, xi=xi)
            def min_modified_ei(x):
                return -self._modified_ei(x.reshape(1, -1), y=y, sense=sense)
            
            min_val = 1
            lb = [val[0] for val in self.space]
            ub = [val[1] for val in self.space]

            for x0 in np.random.uniform(lb, ub, size=(50, len(self.space))):
                if aquisition == 'modified':
                    res = minimize(min_modified_ei, x0=x0, bounds=self.space, method='L-BFGS-B')
                else:
                    res = minimize(min_ei, x0=x0, bounds=self.space, method='L-BFGS-B')
                if res.fun < min_val:
                    min_val = res.fun
                    min_x = res.x               
            return min_x.reshape(1, -1)

    def max_gp_std(self, solver=None):
        if self.classifier is not None:
            return self._constrained_max_gp_std(solver)
        else:
            return self._scipy_max_gp_std()
    
    def exploit_dt(self, x, y, sense='max', solver=None):
        centroids, sizes = self._get_delaunay_centroids_and_sizes(x)
        if sense == 'max':
            index = list(y).index(max(list(y)))
        elif sense == 'min':
            index = list(y).index(min(list(y)))
        exploit_sizes = [0] * len(centroids)
        ind = np.where(np.any(self.delaunay.simplices==index, axis=1))[0]
        for i in ind:
            exploit_sizes[i] = sizes[i]
        return self._dt_milp(centroids, exploit_sizes, solver)
    
    def max_dt(self, x, solver=None):
        # get centroids and sizes
        centroids, sizes = self._get_delaunay_centroids_and_sizes(x)
        return self._dt_milp(centroids, sizes, solver)
    
    def _dt_milp(self, centroids, sizes, solver):
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
        # this is not flexible - a nn classifier outputs needs to pass through sigmoid
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

    def _constrained_modified_ei(self, y, sense, solver):
        if sense == 'max':
            y_opt = np.max(y)
        else:
            y_opt = np.min(y)
        constant_value = self.model.constant_value

        # initialise pyomo model
        m = pyo.ConcreteModel()
        m.mdl = pyo.Block(rule=BlockFormulation(self.model).rule())
        m.mdl_std = pyo.Block(rule=BlockFormulation(self.model).rule(return_std=True))
        m.feas = pyo.Block(rule=BlockFormulation(self.classifier).rule())
        m.n_inputs = set(range(len(self.space)))
        m.inputs = pyo.Var(m.n_inputs, bounds=self.space)
        m.mod_ei = pyo.Var()
        # this is not flexible - a nn classifier outputs needs to pass through sigmoid
        m.feasibility_con = pyo.Constraint(expr= m.feas.outputs[0] >= 0.5 )
        # connect pyomo model input and output to the surrogate models
        m.c = pyo.ConstraintList()
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.mdl.inputs[i] )
            m.c.add( m.inputs[i] == m.mdl_std.inputs[i] )
            m.c.add( m.inputs[i] == m.feas.inputs[i] )

        m.mod_ei_con = pyo.Constraint(
            expr= m.mod_ei ==
            ((constant_value - m.mdl_std.outputs[0]) / (2 * 3.1416)) ** 0.5 * pyo.exp(
                - (y_opt - m.mdl.outputs[0]) ** 2 / (2 * (constant_value - m.mdl_std.outputs[0]))
            )
        )

        m.obj = pyo.Objective(expr=m.mod_ei, sense=pyo.maximize)

        res = solver.solve(m, tee=True)
        x = np.fromiter(m.inputs.extract_values().values(), dtype=float).reshape(1, -1)

        return x
    
    def _ei(self, x, y, sense, xi):
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)

        if sense == 'max':
            y_opt = np.max(y)
        else:
            y_opt = np.min(y)

        with np.errstate(divide='warn'):
            imp = mu - y_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _modified_ei(self, x, y, sense):
        mu, sigma = self.model.predict(x, return_std=True)
        sigma = sigma.reshape(-1, 1)

        if sense == 'max':
            y_opt = np.max(y)
        else:
            y_opt = np.min(y)

        with np.errstate(divide='warn'):
            imp = mu - y_opt
            Z = imp / sigma
            ei = sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
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
