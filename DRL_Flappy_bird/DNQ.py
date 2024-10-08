import torch
from torch import nn
import numpy as np
from copy import deepcopy

class Agent(nn.Module):
    """
    Neural network that represent the agent
    """

    def __init__ (self, state_size: int = 4, action_size: int = 5, load: str = None):
        """ 
        create a NN (RL agent) with state_size neuron as an input and action_size neurons as output
        """
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print("Running on ", self.device)

        # Create a NN whit 1 hidden layers 128 neurons
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_size), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_size)),
        )

        self.model.to(device=self.device)

        # Load preexisting model from parameters
        if load is not None:
            self.load_model(load)

            print("Using old model")


    def forward(self, obs: torch.tensor) -> torch.tensor: 
        """ 
        Forward method, observation as an input and Q_value as an output
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float32, device=self.device)

        return self.model(obs)
    
    def save(self, path:str) -> None: 
        """
        Save model parameters
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path:str):
        self.model.load_state_dict(torch.load(path))
    
class Policy():

    def __init__(self, agent, optimizer, buffer, discount_factor, epsilon, writer = None, target_update_interval=10):
        
        # Define action and target network
        self.action = agent
        self.target = deepcopy(self.action)

        # Store optimizer
        self.optimizer = optimizer

        # Store buffer
        self.buffer = buffer

        # Tensorboard logger
        self.writer = writer

        # Loss algorithm 
        self.loss_fn = torch.nn.MSELoss(reduction="none")

        # Store hyperparam
        self.gamma = discount_factor
        self.eps = epsilon
        self.update_interval = target_update_interval

        # Counter for updating target network
        self.step = 0

    def update_target(self):
        """
        Copy action network weights in target network
        """
        self.target.load_state_dict(self.action.state_dict())
    
    def learn(self):
        """
        Perform a learning step (gradient descent on (yi - Q(s,a))^2 respect to network param)
        """

        # Update target network
        if self.step % self.update_interval == 0:
            self.update_target()

        # Collect a sample from the buffer
        batch = self.buffer.sample()

        q_values = self.action(batch["state"]).cpu()

        actions = torch.as_tensor(batch["action"], dtype=torch.int64).unsqueeze(-1)
        rewards = torch.as_tensor(batch["reward"], dtype=torch.float32).unsqueeze(-1)
        terms = torch.as_tensor(batch["terminated"], dtype=torch.float32).unsqueeze(-1)

        # Get the q_values of the actions selected by the agent
        act_q_values = torch.gather(input=q_values, dim=1, index=actions)

        target_q_values = self.target(batch["next_state"]).cpu()

        max_target_q_values = torch.max(target_q_values, dim=1)[0].unsqueeze(-1)

        # Compute target
        target = rewards + self.gamma*(1-terms)*max_target_q_values

        # Compute loss
        losses = self.loss_fn(act_q_values, target)
        loss = torch.mean(losses)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1


def test_network(NN, enviroment, logger, state, reset):

    # Get q_value from the target network
    q_test = NN(state)

    # Select the action
    action = torch.argmax(q_test)

    # Perform a step in the enviroment
    state, reward, terminated, truncated =  enviroment.step(action)
    ep_reward += reward

    # Update the angle 
    ang = enviroment.ang

    # Reset enviroment if the 
    if terminated or truncated:
        state = enviroment.reset()

        if logger is not None:
            logger.add_scalar("Test/reward", reward, reset)

        ep_reward = 0
        reset += 1

    return ang, state, reset