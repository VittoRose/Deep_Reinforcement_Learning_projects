import torch
from torch import nn

class Actor(nn.Module):
    """
    Neural network used as function approximator:
    Actor   ->  Q(s,a)
    """

    def __init__ (self, state_size:int = 4 , action_size:int = 5, path:str = None):
        """ 
        create a NN (Actor) with state_size neuron as an input and action_size neurons as output
        create a NN (Critic) with state_size neuron as an input and 1 neurons as output
        """
        super().__init__()

        # Store state and action size
        self.state_size = state_size
        self.action_size = action_size
        
        """
        self.model = nn.Sequential(
            nn.Linear(state_size, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
            )
        """
        
        self.model = nn.Sequential(
            nn.Linear(state_size, 128), 
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
            )


        if path is not None:
            self.load(path)

    def forward(self, obs: torch.tensor) -> torch.tensor: 
        """ 
        Forward method, observation as an input and Q_value as an output
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32)

        return self.model(obs)
    
    def save(self, path):
        # Save Actor network after training
        torch.save(self.model.state_dict(), path + "_Actor.pth")

    def load(self, path):
        # Load old parameters from pre-trained network
        self.model.load_state_dict(torch.load(path))

class Critic(nn.Module):
    """
    Neural network used as function approximator:
    Critic   ->  V(s)
    """

    def __init__ (self, state_size:int = 4 , path:str = None):
        """ 
        create a NN (Actor) with state_size neuron as an input and action_size neurons as output
        create a NN (Critic) with state_size neuron as an input and 1 neurons as output
        """
        super().__init__()

        # Store state and action size
        self.state_size = state_size

        # NN as Critic (128 neurons)
        self.model = nn.Sequential(
            nn.Linear(state_size, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

        if path is not None:
            self.load(path)

    def forward(self, obs: torch.tensor) -> torch.tensor: 
        """ 
        Forward method, observation as an input and Q_value as an output
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32)

        return self.model(obs)
    
    def save(self, path):
        # Save Actor network after training
        torch.save(self.model.state_dict(), path + "_Critic.pth")

    def load(self, path):
        # Load old parameters from pre-trained network
        self.model.load_state_dict(torch.load(path))