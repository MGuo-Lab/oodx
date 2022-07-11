import torch
from torch import nn


class NN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.name = 'NN'
        self.activation = activation
        self.layer_nodes = {layer: set(range(nodes)) for layer, nodes in enumerate(layers)}
        self.input_nodes = self.layer_nodes.pop(0)
        self.output_nodes = self.layer_nodes[len(self.layer_nodes)]
        self.activated_nodes = self.layer_nodes.copy()
        self.activated_nodes.pop( len(self.layer_nodes) )
        self.layers = self.build_layers(layers)
        self.network = nn.Sequential(*self.layers)
        self.loss_func = nn.MSELoss()
        self.x_train = None
        self.y_train = None
        self.weights = []
        self.biases = []
    
    def build_layers(self, layers):
        layers_ = []
        for i in range(len(layers) - 2):
            layers_.append( nn.Linear(layers[i], layers[i + 1]) )
            if self.activation == 'tanh':
                layers_.append( nn.Tanh() )
            elif self.activation == 'sigmoid':
                layers_.append( nn.Sigmoid() )
            elif self.activation == 'softplus':
                layers_.append( nn.Softplus() )
            elif self.activation == 'relu':
                layers_.append( nn.ReLU() )
        layers_.append( nn.Linear(layers[-2], layers[-1]) )
        return layers_

    def forward(self, x):
        return self.network(x)
    
    def fit(self, x, y, batch_size=10, epochs=1000, learning_rate=1e-2):
        x_train, y_train = torch.Tensor(x), torch.Tensor(y)
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(epochs):
            permutation = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = permutation[i:i+batch_size]
                x_batch, y_batch = x_train[idx], y_train[idx]    
                predictions = self.forward(x_batch)
                loss = self.loss_func(predictions, y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        
        self.x_train = x
        self.y_train = y
        self.get_params()
    
    def get_params(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.data.numpy())
                self.biases.append(layer.bias.data.numpy())
    
    def predict(self, x):
        x = torch.Tensor(x)
        self.eval()
        y = self.forward(x).detach().numpy()
        return y


class NNClassifier(NN):
    def __init__(self, layers, activation='sigmoid'):
        super().__init__(layers, activation=activation)
        self.name = 'NNClassifier'
    
    def build_layers(self, layers):
        layers_ = []
        for i in range(len(layers) - 1):
            layers_.append( nn.Linear(layers[i], layers[i + 1]) )
            if self.activation == 'sigmoid':
                layers_.append( nn.Sigmoid() )
            elif self.activation == 'hardsigmoid':
                layers_.append(  nn.Hardsigmoid() )
        return layers_


def main():
    net = NN([2, 10, 15, 1], activation='tanh')

    x = np.random.rand(3, 2)
    y = np.random.rand(3, 1) * 10
    print(y)
    net.fit(x, y)
    pred = net.predict(x)
    print(pred)

    classifier = NNClassifier([2, 10, 1])
    t = np.array([[0], [1], [0]])
    classifier.fit(x, t)
    pred = classifier.predict(x)
    print(pred)


if __name__ == '__main__':
    import numpy as np

    main()
