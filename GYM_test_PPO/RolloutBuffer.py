import numpy as np
import torch
from gymnasium import spaces
from parameters import *

class RolloutBuffer():
    """
    :param buffer_size: Max number of element in buffer, correspond to the number of transitions collected in a epoch
    :param state_space: State space form enviroment
    :param action_space: Action space form enviroment
    :param gae_lambda: Gae lambda param, classic advantage if gae_lambda=1
    :param gamma: Discount factor
    :param n_env: Number of parallel enviroment
    """
    def __init__(self, buffer_size: int, state_space: spaces, action_space, gae_lambda: int , gamma: int, n_env: int = 1) -> None:
        
        self.buffer_size = buffer_size
        self.state_space = state_space
        self.action_space = action_space
        self.state_shape = get_state_shape(state_space)
        self.action_shape = get_action_shape(action_space)
        self.n_env = n_env
        
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.reset()

    def reset(self) -> None:
        """
        Reset stored information and buffer properties
        """
        self.state = np.zeros((self.buffer_size, self.n_env, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_env, self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_env), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_env), dtype=np.float32)
        self.ep_start = np.zeros((self.buffer_size, self.n_env), dtype=np.float32)
        self.value = np.zeros((self.buffer_size, self.n_env), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_env), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_env), dtype=np.float32)

        self.index = 0
        self.full = False

    def compute_advantages(self, last_values: torch.tensor, dones: np.ndarray) -> None:
        """
        Compute advantages using gae
        if self.gae_lambda == 1 -> TD(1) -> Monte carlo estimates (sum of discounted rewards)
        else if self.gae_lambda == 0 -> TD(0) -> One step with bootsrapping (r_t + gamma*v(s_{t+1}))
        else -> TD(lambda)
        """

        # Convert last value to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        # Init discounted reward
        last_gae = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_val = last_values
                next_not_terminal = dones.astype(np.float32)
            else: 
                next_not_terminal = 1 - self.ep_start[step+1]
                next_val = self.value[step+1]

        delta = self.rewards[step] + self.gamma*next_val*next_not_terminal - self.value[step]
        last_gae = delta + self.gamma*self.gae_lambda*next_not_terminal*last_gae
        self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def add(self, state:np.ndarray, action:np.ndarray, reward: np.ndarray, ep_start: np.ndarray, value: torch.tensor, log_prob: torch.tensor):
        """
        Add sample to the last avaible index in buffer
        """
        # Reshape 0d tensor
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1,1)

        # Reshape if multiple enviroments
        if isinstance(self.state_space, spaces.Discrete):
            state = state.reshape((self.n_env, *self.state_shape))

        action = action.reshape((self.n_env, self.action_shape))

        self.state[self.index] = np.array(state)
        self.actions[self.index] = np.array(action)
        self.rewards[self.index] = np.array(reward)
        self.ep_start[self.index] = np.array(ep_start)
        self.value[self.index] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.index] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True



def get_state_shape(state_space: spaces):
    """
    Get the state space dimension from a gym enviroment
    """

    if isinstance(state_space, spaces.Box):
        return state_space.shape
    elif isinstance(state_space, spaces.Discrete):
        return (1,)
    elif isinstance(state_space, spaces.MultiDiscrete):
        return(int(len(state_space.nvec)),)
    else:
        raise NotImplementedError("Enviroment not supported")
    
def get_action_shape(action_space: spaces):
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(len(action_space.nvec))
    else:
        raise NotImplementedError("Enviroment not supported")

    
if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    print(get_state_shape(env.observation_space))
    print(get_action_shape(env.action_space))

    buffer = RolloutBuffer(BUFFER_SIZE, env.observation_space, env.action_space, gae_lambda=LAMBDA, gamma=GAMMA)

    print("Stat",buffer.state)
    print("Act",buffer.actions)