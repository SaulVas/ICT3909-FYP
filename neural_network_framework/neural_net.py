import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rates):
        """
        A flexible feed-forward neural network with configurable architecture.

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output (default: 1 for regression)
            hidden_layers (list): List of integers representing the size of each hidden layer
            dropout_rates (list): List of dropout rates for each layer
        """
        super(FeedForwardNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rates = dropout_rates
        self.output_dim = output_dim

        # Create network layers dynamically
        layers = []

        # Input layer to first hidden layer
        prev_dim = input_dim

        # Hidden layers
        for _, (hidden_dim, dropout_rate) in enumerate(
            zip(hidden_layers, dropout_rates)
        ):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
