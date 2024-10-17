import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class Agent(nn.Module):
    """
    Create a ActorCritic agent 
    :param envs: Gymnasium SyncVectorEnv, training enviroment
    :param init_layer: flag for initializing with weight and biases defined in layer_init
    """
    def __init__(self, envs, init_layer: bool = True):

        super(Agent, self).__init__()

        # Using Tanh as activation function as suggested in various implementation
        self.critic = make_critic(envs, init_layer)
        
        self.actor = make_actor(envs, init_layer)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_critic(envs, init_layer) -> torch.nn:
    """
    Function to create critic network, modify here the network structure
    """
    if init_layer:
        critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
            )
    else:
        critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            )
    return critic

def make_actor(envs, init_layer) -> torch.nn:

    """
    Function to create actor network, modify here the network structure
    """

    if init_layer:
        actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    else:
        actor = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n)
        )
    return actor