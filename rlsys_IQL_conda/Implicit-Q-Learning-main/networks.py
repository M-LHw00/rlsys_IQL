import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_dim1, action_dim2, hidden_size=32, init_w=3e-3, log_std_min=-10, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head1 = nn.Linear(hidden_size,action_dim1)
        self.head2 = nn.Linear(hidden_size,action_dim2)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        a1 = self.head1(x)
        a2 = self.head1(x)
        return a1, a2
    


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_dim1, action_dim2, hidden_size=32, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head1 = nn.Linear(hidden_size,action_dim1)
        self.head2 = nn.Linear(hidden_size,action_dim2)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v1 = self.head1(x)
        v2 = self.head1(x)
        return v1, v2
    
class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, hidden_size=32):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)



    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)