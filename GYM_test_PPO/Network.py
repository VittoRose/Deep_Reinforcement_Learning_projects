import torch
from torch import nn
from torch.distributions.categorical import Categorical

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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64), 
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
            )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64,1)
            )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_value(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        action = probs.sample()

        return action, probs.log_prob(action), probs.entropy, self.critic(x)


    """
    def forward(self, state: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        return self.forward_actor(state), self.forward_critic(state)
    
    def forward_actor(self, state) -> torch.tensor:
        return self.policy(state)
    
    def forward_critic(self, state) -> torch.tensor:
        return self.value(state)
    """
    
if __name__ == "__main__":
    ac = ActorCritic(2, 2)

    action, log_prob, entr, value = ac.get_action_value(torch.rand(2))

    print("Action ", action)
    print("Value ", value)
    print("entropy ", entr)
    print("logprob ", log_prob)