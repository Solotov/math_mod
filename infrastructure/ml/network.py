import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int = 1, activation: str = "relu"):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self, method="xavier"):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)