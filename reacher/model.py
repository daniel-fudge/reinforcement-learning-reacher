import torch
import torch.nn as nn
import torch.nn.functional as nn_f


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h_sizes):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed.
            h_sizes (tuple): The number of nodes for each hidden layer.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = zip(h_sizes[:-1], h_sizes[1:])
        
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, h_sizes[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(h_sizes[-1], action_size)

    def forward(self, x):
        """The forward propagation method.

        Args:
            x (array_like): The current state.

        Returns:
            torch.tensor:  The final output of the network.
        """
        for linear in self.hidden_layers:
            x = nn_f.relu(linear(x))

        return self.output(x)
