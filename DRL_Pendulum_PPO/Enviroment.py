import random
import numpy as np
import torch
from parameters import *

class PendolumEnv:
    """
    Enviroment for DRL problem
    """

    def __init__(self, l=1, m=1, g=9.81, dt=1/60):
        self.l = l
        self.m = m
        self.g = g
        self.dt = dt

        self.env_reset = 0
        self.obs_size = 4
        self.action_size = 5

        self.reset()

    def reset(self) -> torch.tensor:
        """
        Reset the enviroment with random initial condition
        """
        
        self.theta = random.uniform(-np.pi/2+0.2, np.pi/2-0.2)
        self.omega = random.uniform(-1.5,1.5)

        # Lenght of episode
        self.episode_len = 0
        
        self.env_reset += 1

        return self.get_state()
    
    @property
    def ang(self):
        """
        Function that return the current angle in the enviroment
        """
        return self.theta
    
    def step(self, action : int) -> tuple[torch.tensor, int, bool, bool]:
        """
        Perform a given action to the enviroment
        """

        # Init variable
        reward = 0
        terminated = False
        truncated = False

        # Decode the NN output in torque value
        match action:
            case 0:
                tau = -50
            case 1:
                tau = -1
            case 2: 
                tau = 0
            case 3:
                tau = 1
            case 4: 
                tau = 50
            case _:
                raise ValueError("Action non consistent")

        # Dynamic equation
        alpha = (-self.g / self.l) * np.sin(self.theta) + tau / (self.m * self.l**2)
        
        # Euler to update state variable
        self.omega = self.omega + alpha * self.dt
        self.theta = self.theta + self.omega * self.dt

        # Increase episode length
        self.episode_len += 1

        # Evaluate reward
        if self.theta > MAX_ANGLE or self.theta < -MAX_ANGLE:
            terminated = True
            reward += MAX_ANGLE_EXCEEDED

        if -SMALL_ANGLE < self.theta < SMALL_ANGLE:
            reward += SMALL_ANGLE_REWARD

        if self.episode_len > 1000:
            reward += 2
            terminated = True


        return (self.get_state(), reward, terminated, truncated)
    
    def get_state(self) -> torch.tensor:
        """
        Return the enviroment state as 4 number between 0 and 1, as a tensor type
        """
        
        # Get sign of angle and velocity
        ang_sign = 1 if self.theta > 0 else 0
        vel_sign = 1 if self.omega > 0 else 0

        # Normalize angle and velocity
        theta_in = (self.theta - (-np.pi))/(np.pi - (-np.pi)) 
        omega_in = (self.omega - (-MAX_OMEGA)/(MAX_OMEGA - (-MAX_OMEGA)))

        return torch.as_tensor([ang_sign, theta_in, vel_sign, omega_in], dtype=torch.float32)
    

class EnvWorker:
    """
    Environment worker class. It represents a single environment.
    """
    def __init__(self, env):
        """
        EnvWorker initializer.
        :param env: Environment class object.
        """
        self.env = env
        # Result stores the information received from the environment
        self.result = None

    def reset(self):
        """
        Reset the environment.
        :param kwargs: Environment keyword arguments.
        """
        # Get observation from environment
        obs = self.env.reset()
        self.result = obs

    def send(self, action : list[int]):
        """
        Perform an action in the environment.
        :param action: Action to apply into the environment.
        """
        next_state, reward, terminated, truncated = self.env.step(action)
        self.result = (next_state, reward, terminated, truncated)

    def receive(self) -> tuple[torch.tensor, int, bool, bool]:
        """
        Return the information received from the environment.
        :return: Tuple of environment data.
        return only a torch.tensor if recive() is called after reset()
        otherwise it return -> tuple[torch.tensor, int, bool, bool]
        """
        return self.result
    

class VectorEnv():
    """
    Pool of enviroment
    """
    def __init__(self, env: list):

        self.envs = env
        self.workers = [EnvWorker(i) for i in env]

    def __len__(self) -> int:
        """
        Return the number of environments.
        :return: The number of the environments.
        """
        return len(self.envs)
    
    def reset(self, indices=None) -> np.array:
        """
        Reset the all or the specified environments.
        :param indices: Multiple environment indices must be given as an iterable. A single
        environment index can also be provided as a scalar. Passing None means all the environments. Default to None.
        :param kwargs: Environment keyword arguments.
        :return: Stacked environment observations.
        """
        # Get an iterable of environment indices
        if indices is None:
            # Every environment
            workers = list(range(len(self.envs)))
        elif np.isscalar(indices):
            # A single environment index is given as input
            # Convert to a list with a single index
            workers = [indices]
        else:
            # The indices are already given as an iterable, e.g. list, numpy array
            workers = indices

        for index in workers:
            # Reset the single environment
            self.workers[index].reset()
        # Stack every received observation by the EnvWorkers
        obses = [self.workers[index].receive() for index in workers]            # List of (#index) array (type = [torch.tensor,])
        return np.stack(obses)                                                  # Matrix (#index)_rows x (state_size)_col
    
    def step(self, actions, indices=None):
        """
        Perform actions into all or the specified environments.
        :param actions: List of actions to perform.
        :param indices: Multiple environment indices must be given as an iterable. A single
        environment index can also be provided as a scalar. Passing None means all the environments. Default to None.
        :return: Stacked environment responses.
        """

        if indices is None:
            # Every environment
            workers = list(range(len(self.envs)))
        elif np.isscalar(indices):
            # A single environment index is given as input
            # Convert to a list with a single index
            workers = [indices]
        else:
            # The indices are already given as an iterable, e.g. list, numpy array
            workers = indices

        # Perform a step into every EnvWorker
        for i, j in enumerate(workers):
            self.workers[j].send(actions[i])

        # Prepare the results
        result = []
        for j in workers:
            # Get environment step return
            env_return = self.workers[j].receive()
            result.append(env_return)

        # Unpack result        
        next_obses, rewards, terms, truncs = tuple(zip(*result))
        
        return (
            np.stack(next_obses),
            np.stack(rewards),
            np.stack(terms),
            np.stack(truncs)) 
