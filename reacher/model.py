import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_f

# Set Architecture hyperparameters
FC1_UNITS = 128          # Size if 1st Hidden layer
FC2_UNITS = 64           # Size if 2nd Hidden layer


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


# noinspection DuplicatedCode
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS, FC2_UNITS)
        self.fc3 = nn.Linear(FC2_UNITS, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = nn_f.relu(self.fc1(state))
        x = nn_f.relu(self.fc2(x))
        return nn_f.tanh(self.fc3(x))


# noinspection DuplicatedCode
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS + action_size, FC2_UNITS)
        self.fc3 = nn.Linear(FC2_UNITS, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fcs1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = nn_f.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = nn_f.relu(self.fc2(x))
        return self.fc3(x)
