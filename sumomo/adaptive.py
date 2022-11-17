import numpy as np
from numpy.linalg import det
from scipy.spatial import Delaunay
import pyomo.environ as pyo
import math
import itertools

from .formulations import SumoBlock


class AdaptiveSampler:
    def __init__(self, space):
        self.space = space
        self.delaunay = None

    def max_gp_std(self, model):
        ''' maximise a Gaussian process regression standard deviation in predictions
            this is an exploration only adaptive sampling method
        '''
        m = pyo.ConcreteModel()
        
        block = SumoBlock(model)
        m.mdl = block.get_formulation(return_std=True)
        
        m.n_inputs = set(range(len(self.space)))
        m.inputs = pyo.Var(m.n_inputs, bounds=self.space)
        m.obj = pyo.Objective(
            expr=m.mdl.outputs[0], 
            sense=pyo.maximize)
        m.c = pyo.ConstraintList()
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.mdl.inputs[i] )
        return m

    #def max_constrained_gp_std(self):
        ''' same as max_gp_std with additional feasibility 
            constraints from classifier
        '''
        m = self.max_gp_std()
        
        block = SumoBlock(self.classifier)
        m.feas = block.get_formulation()

        if self.classifier.name == 'NN':
            m.feasibility_con = pyo.Constraint(
                expr= 
                1 / (1 + pyo.exp(-m.feas.outputs[0])) >= 0.5
            )
        else:
            m.feasibility_con = pyo.Constraint(
                expr= m.feas.outputs[0] >= 0.5
            )
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.feas.inputs[i] )
        return m
    
    def max_triangle(self, x, include_vertices=0):
        ''' choose maximum sized region from Delaunay triangulation
            this is an exploration only adaptive sampling method
        '''
        centroids, sizes = self._get_delaunay_centroids_and_sizes(x, include_vertices)
        return self._delaulay_triangle_milp(centroids, sizes)

    # def max_constrained_triangle(self, x):
        ''' same as max_triangle with additional 
            feasibility constraints from classifier
        '''
        m = self.max_triangle(x)
        
        block = SumoBlock(self.classifier)
        m.feas = block.get_formulation()

        if self.classifier.name == 'NN':
            m.feasibility_con = pyo.Constraint(
                expr= 
                1 / (1 + pyo.exp(-m.feas.outputs[0])) >= 0.5
            )
        else:
            m.feasibility_con = pyo.Constraint(
                expr= 
                m.feas.outputs[0] >= 0.5 
            )
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.feas.inputs[i] )
        return m

    def modified_expected_improvement(self, model, y, sense):
        ''' maximise modified expected improvement of 
            Gaussian process regression model
            this method addresses the exploration/exploitation trade-off
        '''
        if sense == 'max':
            y_opt = np.max(y)
        else:
            y_opt = np.min(y)
        constant_value = model.constant_value
        m = pyo.ConcreteModel()
   
        block = SumoBlock(model)
        m.mdl = block.get_formulation()
        m.mdl_std = block.get_formulation(return_std=True)

        m.n_inputs = set(range(len(self.space)))
        m.inputs = pyo.Var(m.n_inputs, bounds=self.space)
        m.mod_ei = pyo.Var()
        m.c = pyo.ConstraintList()
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.mdl.inputs[i] )
            m.c.add( m.inputs[i] == m.mdl_std.inputs[i] )
        m.mod_ei_con = pyo.Constraint(
            expr= m.mod_ei == pyo.sqrt(
                (constant_value + m.mdl_std.outputs[0]) / (2 * 3.1416)) * pyo.exp(
                    -(y_opt - m.mdl.outputs[0]) ** 2 / (2 * (constant_value + m.mdl_std.outputs[0]))
            )
        )
        m.obj = pyo.Objective(expr=m.mod_ei, sense=pyo.maximize)
        return m

    # def constrained_modified_expected_improvement(self, y, sense):
        ''' same as modified_expected_improvement with 
            additional feasibility constraints from classifier
        '''
        m = self.modified_expected_improvement(y, sense)

        block = SumoBlock(self.classifier)
        m.feas = block.get_formulation()
        
        if self.classifier.name == 'NN':
            m.feasibility_con = pyo.Constraint(
                expr= 
                1 / (1 + pyo.exp(-m.feas.outputs[0])) >= 0.5
            )
        else:
            m.feasibility_con = pyo.Constraint(
                expr= 
                m.feas.outputs[0] >= 0.5 
            )
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.feas.inputs[i] )
        return m

    def exploit_triangle(self, x, y, sense, include_vertices=0):
        ''' chooses maximum sized region from Delauanay 
            triangulation connected to min/max sample
            this is an exploitation only adaptive sampling method
        '''
        centroids, sizes = self._get_delaunay_centroids_and_sizes(x, include_vertices)
        if sense == 'max':
            index = list(y).index(max(list(y)))
        elif sense == 'min':
            index = list(y).index(min(list(y)))
        exploit_sizes = [0] * len(centroids)
        ind = np.where(np.any(self.delaunay.simplices==index, axis=1))[0]
        for i in ind:
            exploit_sizes[i] = sizes[i]
        return self._delaulay_triangle_milp(centroids, exploit_sizes)

    # def exploit_constrained_triangle(self, x, y, sense):
        ''' same as exploit_triangle with additional 
            feasibility constraints from classifier
        '''
        m = self.exploit_triangle(x, y, sense)

        block = SumoBlock(self.classifier)
        m.feas = block.get_formulation()

        if self.classifier.name == 'NN':
            m.feasibility_con = pyo.Constraint(
                expr= 
                1 / (1 + pyo.exp(-m.feas.outputs[0])) >= 0.5
            )
        else:
            m.feasibility_con = pyo.Constraint(
                expr= 
                m.feas.outputs[0] >= 0.5 
            )
        for i in m.n_inputs:
            m.c.add( m.inputs[i] == m.feas.inputs[i] )
        return m

    def _get_delaunay_centroids_and_sizes(self, x, include_vertices=0):
        for i, bounds in enumerate(self.space):
            x = x[x[:, i] >= bounds[0]]
            x = x[x[:, i] <= bounds[1]]
        
        if include_vertices:
            vertices = np.array(list(itertools.product(*self.space)))
            print(vertices)

        print(x)
        
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
    
    def _delaulay_triangle_milp(self, centroids, sizes):
        m = pyo.ConcreteModel()
        m.n_dt = set(range(len(centroids)))
        m.y = pyo.Var(m.n_dt, domain=pyo.Binary)
        m.max_dt = pyo.Var()
        m.n_inputs = set(range(len(self.space)))
        m.inputs = pyo.Var(m.n_inputs, bounds=self.space)
        m.exactly_one_con = pyo.Constraint(
            expr= 
            sum(m.y[i] for i in m.n_dt) == 1 
        )
        m.c = pyo.ConstraintList()
        for i in m.n_inputs:
            m.c.add(
                m.inputs[i] == sum(centroids[k, i] * m.y[k] for k in m.n_dt)
            )
        m.obj = pyo.Objective(
            expr=sum(m.y[i] * sizes[i] for i in m.n_dt), 
            sense=pyo.maximize)
        return m
