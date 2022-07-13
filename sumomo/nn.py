import torch
from torch import nn


class NN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.name = 'NN'
        self.activation = activation
        self.layers = self._build_layers(layers)
        self.network = nn.Sequential(*self.layers)
        self.weights = []
        self.biases = []

    def forward(self, x):
        return self.network(x)
    
    def fit(self, x, y, batch_size=10, epochs=1000, learning_rate=1e-2, loss_func=nn.MSELoss()):
        x_train, y_train = torch.Tensor(x), torch.Tensor(y)
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(epochs):
            permutation = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = permutation[i:i+batch_size]
                x_batch, y_batch = x_train[idx], y_train[idx]    
                predictions = self.forward(x_batch)
                loss = loss_func(predictions, y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        self._get_params()
    
    def predict(self, x):
        x = torch.Tensor(x)
        self.eval()
        y = self.forward(x).detach().numpy()
        return y
    
    def _get_params(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.data.numpy())
                self.biases.append(layer.bias.data.numpy())
    
    def _activation_selector(self):
        if self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'softplus':
            return nn.Softplus()
        elif self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'hardsigmoid':
            return nn.Hardsigmoid()

    def _build_layers(self, layers):
        torch_layers = []
        for i in range(len(layers) - 2):
            torch_layers.append( nn.Linear(layers[i], layers[i + 1]) )
            torch_layers.append( self._activation_selector() )
        torch_layers.append( nn.Linear(layers[-2], layers[-1]) )
        return torch_layers


class NNClassifier(NN):
    def __init__(self, layers, activation='sigmoid'):
        super().__init__(layers, activation=activation)
        self.name = 'NNClassifier'
    
    def _build_layers(self, layers):
        torch_layers = []
        for i in range(len(layers) - 1):
            torch_layers.append( nn.Linear(layers[i], layers[i + 1]) )
            torch_layers.append( self._activation_selector() )
        return torch_layers
