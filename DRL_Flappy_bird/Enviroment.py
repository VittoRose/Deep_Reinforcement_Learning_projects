from Jumping_bird import *
import numpy as np
import torch
from parameters import *

class FlappyBird:
    """
    Enviroment for DRL problem
    """

    def __init__(self):

        # Network size
        self.state_size = 4
        self.action_size = 2

        # Init bird and obstacle
        self.bird = None
        self.pipes = None

        # Init distance 
        self.dist = PIPE_X0

        # Stats param
        self.env_reset = -1
        self.score = 0

        self.reset()

    def reset(self) -> torch.tensor:
        """
        Reset the enviroment with random initial condition
        """

        self.env_reset += 1
        self.score = 0

        t0 = time.time()

        if self.bird is not None and self.pipes is not None:
            del self.bird, self.pipes

        self.bird = Bird(r=10, g=300)
        self.pipes = [Obstacle(PIPE_X0 + 2*ORIZONTAL_GAP*i, t0) for i in range(PIPE_ON_SCREEN)] 

        self.dist = PIPE_X0     

        return self.get_state()
        
    def step(self, action : int) -> tuple[torch.tensor, int, bool, bool]:
        """
        Perform a given action to the enviroment
        """

        # Output variable preallocation
        reward = 0
        terminated = False
        truncated = False

        # Decode action to game input
        if 0 == action:
            jump = False
        elif 1 == action:
            jump = True
        else:
            raise ValueError("Action not allowed")

        # Game loop
        self.bird.update(jump)

        # Move every pipe
        for pipe in self.pipes:
            pipe.update()

            # Reset pipe when out of screen
            if pipe.x < -pipe.width:
                pipe.reset()

        # Evaluate the min distance from pipe and update score
        
        self.dist, self.score, index = distance_score(self.bird, self.pipes, self.score)

        # Reward the agent if the bird is headin to a pipe
        if entering_pipe(self.bird, self.pipes, index):
            reward += HEADING_TO_PIPE

        # Reward the agent if the bird is between pipe
        for pipe in self.pipes:
            if pipe.x < self.bird.x < pipe.x + pipe.width:
                reward += BETWEEN_PIPE

        # Collision with pipe 
        if collision_detect(self.bird, self.pipes):
            terminated = True
            reward = COLLISION_REW
            self.env_reset += 1
            self.reset()
        #else:
         #   reward += ALIVE_REW

        return self.get_state(), reward, terminated, truncated
    
    def get_disp_data(self) -> tuple[Bird, Obstacle ]:
        """
        Return the necessary data to draw the enviroment
        """

        return self.bird, self.pipes

        
    
    def get_state(self) -> torch.tensor:
        """
        Return the enviroment state as 4 number between 0 and 1, as a tensor type
        """

        # Normalized bird y 
        y = self.bird.y/HEIGHT

        # Normalized velocity
        vy = self.bird.vy/MAX_VEL

        # Distance to the next pipe
        d_out = self.dist/WIDTH

        # Normalized pipe velocity
        vp_x = self.pipes[0].vx/MAX_PIPE_SPEED

        # Pipe width
        w = self.pipes[0].width/WIDTH

        return torch.as_tensor([y, vy, d_out, vp_x], dtype=torch.float32)
    

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
        next_obs, reward, terminated, truncated = self.env.step(action)
        self.result = (next_obs, reward, terminated, truncated)

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
