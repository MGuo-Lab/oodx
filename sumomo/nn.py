import torch
from torch import nn


class NN(nn.Sequential):
    def __init__(self, layers, activation='tanh'):
        self.name = 'NN'
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []
        super().__init__(*self._build_layers(layers))
    
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
    
    def predict(self, x, return_proba=False, return_class=False, threshold=0.5):
        x = torch.Tensor(x)
        self.eval()
        y = self.forward(x).detach()
        c = torch.max(y, torch.tensor([0.]))
        proba = torch.sigmoid(y).detach()
        c[proba > threshold] = 1
        if return_class and return_proba:
            return y.numpy(), proba.numpy(), c.numpy()
        elif return_class:
            return y.numpy(), c.numpy()
        elif return_proba:
            return y.numpy(), proba.numpy()
        else:
            return y.numpy()
    
    def _get_params(self):
        for layer in self:
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
    