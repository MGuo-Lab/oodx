import pyomo.environ as pyo


class BlockFormulation:
    def __init__(self, model, bounds):
        self.model = model
        self.bounds = bounds
    
    def get_model(self):
        if self.model.name == 'NN':
            if self.model.activation == 'relu':
                return self._nn_relu_rule
            if self.model.activation == 'tanh':
                return self._nn_tanh_rule
            if self.model.activation == 'softplus':
                return self._nn_softplus_rule
            if self.model.activation == 'sigmoid':
                return self._nn_sigmoid_rule
        
        if self.model.name == 'NNClassifier':
            if self.model.activation == 'sigmoid':
                return self._nn_sigmoid_classifier_rule
            if self.model.activation == 'hardsigmoid':
                return self._nn_hardsigmoid_rule
        
        if self.model.name == 'GPR':
            return self._gpr_rule
        
        if self.model.name == 'GPC':
            return self._gpc_rule

    def _gpr_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.length_scale
        constant_value = self.model.constant_value
        alpha = self.model.alpha

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))
        
        # declare variables
        m.inputs = pyo.Var(n_inputs, bounds=self.bounds)
        m.output = pyo.Var()

        # gpr constraint        
        m.gpr = pyo.Constraint(expr=
            m.output == 
            sum(alpha[i] * constant_value * pyo.exp(-sum(0.5 / length_scale ** 2 * 
            (m.inputs[j] - x_train[i, j]) ** 2 for j in n_inputs)) 
            for i in n_samples)
        )
    
    def _nn_non_activated_output(self, m):
        # declare parameters
        W = self.model.weights
        b = self.model.biases
        
        # declare sets
        layer_nodes = self.model.layer_nodes
        m.input_nodes = self.model.input_nodes
        m.output_nodes = self.model.output_nodes
        activated_nodes = self.model.activated_nodes
        nodes = set([(i, j) for i in layer_nodes.keys() for j in layer_nodes[i]])
        m.a_nodes = set([(i, j) for i in activated_nodes.keys() for j in activated_nodes[i]])
        
        # declare variables
        m.inputs = pyo.Var(m.input_nodes, bounds=self.bounds)
        m.z = pyo.Var(nodes)
        m.a = pyo.Var(m.a_nodes)
        m.outputs = pyo.Var(m.output_nodes)

        # constraints
        m.c = pyo.ConstraintList()

        for l, n in nodes:
            # first hidden layer receives inputs
            if l == 1:
                m.c.add(
                    m.z[(l, n)] == sum(W[l - 1][n, k] * m.inputs[k] for k in m.input_nodes) + b[l - 1][n]
                )
            # all other layers receive previous layer activated outputs
            if l > 1:
                m.c.add(
                    m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in layer_nodes[l - 1]) + b[l - 1][n]
                )
            # output layer returns outputs of linear function
            if l == len(layer_nodes):
                m.c.add( m.outputs[n] == m.z[(l, n)] )


    def _nn_tanh_rule(self, m):
        # retrieve general nn model
        m.nn = pyo.Block(rule=self._nn_non_activated_output)
    
        # declare variables
        m.inputs = pyo.Var(m.nn.input_nodes, bounds=self.bounds)
        m.outputs = pyo.Var(m.nn.output_nodes)

        # constraints
        m.c = pyo.ConstraintList()
      
        # activated layers return tanh of linear outputs
        for l, n in m.nn.a_nodes:
            m.c.add( m.nn.a[(l, n)] == 1 - 2 / (pyo.exp(2 * m.nn.z[(l, n)]) + 1) )
        
        # connect inputs to general model
        for i in m.nn.input_nodes:
            m.c.add( m.inputs[i] == m.nn.inputs[i] )
        
        # connect outputs to general model
        for i in m.nn.output_nodes:
            m.c.add( m.outputs[i] == m.nn.outputs[i] )


    def _nn_sigmoid_rule(self, m):
        # retrieve general nn model
        m.nn = pyo.Block(rule=self._nn_non_activated_output)
    
        # declare variables
        m.inputs = pyo.Var(m.nn.input_nodes, bounds=self.bounds)
        m.outputs = pyo.Var(m.nn.output_nodes)

        # constraints
        m.c = pyo.ConstraintList()
      
        # activated layers return sigmoid of linear outputs
        for l, n in m.nn.a_nodes:
            m.c.add( m.nn.a[(l, n)] == 1 / (1 + pyo.exp(-m.nn.z[(l, n)])) )
        
        # connect inputs to general model
        for i in m.nn.input_nodes:
            m.c.add( m.inputs[i] == m.nn.inputs[i] )
        
        # connect outputs to general model
        for i in m.nn.output_nodes:
            m.c.add( m.outputs[i] == m.nn.outputs[i] )
    

    def _nn_softplus_rule(self, m):
        # retrieve general nn model
        m.nn = pyo.Block(rule=self._nn_non_activated_output)
    
        # declare variables
        m.inputs = pyo.Var(m.nn.input_nodes, bounds=self.bounds)
        m.outputs = pyo.Var(m.nn.output_nodes)

        # constraints
        m.c = pyo.ConstraintList()
      
        # activated layers return softplus of linear outputs
        for l, n in m.nn.a_nodes:
            m.c.add( m.nn.a[(l, n)] == pyo.log(1 + pyo.exp(m.nn.z[(l, n)])) )
        
        # connect inputs to general model
        for i in m.nn.input_nodes:
            m.c.add( m.inputs[i] == m.nn.inputs[i] )
        
        # connect outputs to general model
        for i in m.nn.output_nodes:
            m.c.add( m.outputs[i] == m.nn.outputs[i] )


    def _nn_relu_rule(self, m):
        # retrieve general nn model
        m.nn = pyo.Block(rule=self._nn_non_activated_output)
    
        # declare variables
        m.inputs = pyo.Var(m.nn.input_nodes, bounds=self.bounds)
        m.outputs = pyo.Var(m.nn.output_nodes)
        m.y = pyo.Var(m.nn.a_nodes, domain=pyo.Binary)

        # constraints
        m.c = pyo.ConstraintList()
      
        # activated layers return ReLU of linear outputs, big-M formulation
        for l, n in m.nn.a_nodes:
            m.c.add( m.nn.a[(l, n)] >= 0 )
            m.c.add( m.nn.a[(l, n)] >= m.nn.z[(l, n)] )
            m.c.add( m.nn.a[(l, n)] <= 1e6 * m.y[(l, n)] )
            m.c.add( m.nn.a[(l, n)] <= m.nn.z[(l, n)] + 1e6 * (1 - m.y[(l, n)]) )
        
        # connect inputs to general model
        for i in m.nn.input_nodes:
            m.c.add( m.inputs[i] == m.nn.inputs[i] )
        
        # connect outputs to general model
        for i in m.nn.output_nodes:
            m.c.add( m.outputs[i] == m.nn.outputs[i] )


    def _nn_activated_output(self, m):
        # declare parameters
        W = self.model.weights
        b = self.model.biases
        
        # declare sets
        layer_nodes = self.model.layer_nodes
        m.input_nodes = self.model.input_nodes
        m.output_nodes = self.model.output_nodes
        m.nodes = set([(i, j) for i in layer_nodes.keys() for j in layer_nodes[i]])
        
        # declare variables
        m.inputs = pyo.Var(m.input_nodes, bounds=self.bounds)
        m.z = pyo.Var(m.nodes)
        m.a = pyo.Var(m.nodes)
        m.outputs = pyo.Var(m.output_nodes)

        # constraints
        m.c = pyo.ConstraintList()

        for l, n in m.nodes:
            # first hidden layer receives inputs
            if l == 1:
                m.c.add(
                    m.z[(l, n)] == sum(W[l - 1][n, k] * m.inputs[k] for k in m.input_nodes) + b[l - 1][n]
                )
            # all other layers receive previous layer activated outputs
            if l > 1:
                m.c.add(
                    m.z[(l, n)] == sum(W[l - 1][n, k] * m.a[(l - 1, k)] for k in layer_nodes[l - 1]) + b[l - 1][n]
                )
            # output layer returns outputs of activation function
            if l == len(layer_nodes):
                m.c.add( m.outputs[n] == m.a[(l, n)] )


    def _nn_sigmoid_classifier_rule(self, m):
        # retrieve general nn model
        m.nn = pyo.Block(rule=self._nn_activated_output)
    
        # declare variables
        m.inputs = pyo.Var(m.nn.input_nodes, bounds=self.bounds)
        m.outputs = pyo.Var(m.nn.output_nodes)

        # constraints
        m.c = pyo.ConstraintList()
      
        # activated layers return sigmoid of linear outputs
        for l, n in m.nn.nodes:
            m.c.add( m.nn.a[(l, n)] == 1 / (1 + pyo.exp(-m.nn.z[(l, n)])) )
        
        # connect inputs to general model
        for i in m.nn.input_nodes:
            m.c.add( m.inputs[i] == m.nn.inputs[i] )
        
        # connect outputs to general model
        for i in m.nn.output_nodes:
            m.c.add( m.outputs[i] == m.nn.outputs[i] )


    def _nn_hardsigmoid_rule(self, m):
        # retrieve general nn model
        m.nn = pyo.Block(rule=self._nn_activated_output)
    
        # declare variables
        m.inputs = pyo.Var(m.nn.input_nodes, bounds=self.bounds)
        m.outputs = pyo.Var(m.nn.output_nodes)
        m.y0 = pyo.Var(m.nn.nodes, domain=pyo.Binary)
        m.y1 = pyo.Var(m.nn.nodes, domain=pyo.Binary)

        # constraints
        m.c = pyo.ConstraintList()
      
        # activated layers return HardSigmoid of linear outputs, big-M formulation
        for l, n in m.nn.nodes:
            m.c.add( m.nn.a[(l, n)] <= m.y0[(l, n)] )
            m.c.add(
                m.nn.a[(l, n)] >= 
                m.nn.z[(l, n)] / 6 + 0.5 - 1e6 * (1 - m.y0[(l, n)] + m.y1[(l, n)])
            )
            m.c.add(
                m.nn.a[(l, n)] <= 
                m.nn.z[(l, n)] / 6 + 0.5 + 1e6 * (1 - m.y0[(l, n)] + m.y1[(l, n)])
            )
            m.c.add( m.nn.a[(l, n)] >= m.y1[(l, n)] )
            m.c.add( m.nn.z[(l, n)] - 1e6 * m.y0[(l, n)] <= -3 )
            m.c.add( m.nn.z[(l, n)] + 1e6 * (1 - m.y0[(l, n)]) >= -3 )
            m.c.add( m.nn.z[(l, n)] - 1e6 * m.y1[(l, n)] <= 3 )
            m.c.add( m.nn.z[(l, n)] + 1e6 * (1 - m.y1[(l, n)]) >= 3 )
        
        # connect inputs to general model
        for i in m.nn.input_nodes:
            m.c.add( m.inputs[i] == m.nn.inputs[i] )
        
        # connect outputs to general model
        for i in m.nn.output_nodes:
            m.c.add( m.outputs[i] == m.nn.outputs[i] )

    
    def _gpc_rule(self, m):
        # declare parameters
        x_train = self.model.x_train
        length_scale = self.model.l
        constant_value = self.model.sigma_f ** 2
        delta = self.model.delta
        invP = self.model.invP

        # declare sets
        n_samples = set(range(x_train.shape[0]))
        n_inputs = set(range(x_train.shape[1]))

        # declare variables
        m.inputs = pyo.Var(n_inputs, bounds=self.bounds)
        m.output = pyo.Var()

        # gpc constraint
        m.gpc = pyo.Constraint(expr=
            m.output == 
            1 / (1 + pyo.exp(-constant_value * sum(delta[j] * pyo.exp(-sum(0.5 / length_scale ** 2 * (m.inputs[i] - x_train[j, i]) ** 2 for i in n_inputs)) for j in n_samples) / pyo.sqrt(1 + 3.1416 / 8 * constant_value * (1 - sum(pyo.exp(-sum(0.5 / length_scale ** 2 * (m.inputs[i] - x_train[j, i]) ** 2 for i in n_inputs)) * constant_value * sum(pyo.exp(-sum(0.5 / length_scale ** 2 * (m.inputs[j] - x_train[k, j]) ** 2 for j in n_inputs)) * invP[j, k] for k in n_samples) for j in n_samples)))))
        )
