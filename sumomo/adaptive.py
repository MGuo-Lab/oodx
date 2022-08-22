import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import pyomo.environ as pyo

from .formulations import BlockFormulation


class AdaptiveSampler:
    def __init__(self, space, model, classifier=None):
        self.space = space
        self.model = model
        self.classifier = classifier

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

    def max_gp_std(self, solver=None):
        if self.classifier is not None:
            return self._constrained_max_gp_std(solver)
        else:
            return self._scipy_max_gp_std()
