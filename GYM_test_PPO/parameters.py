
# Training parameters
n_env = 4
n_step = 128                    # Number of step in the enviroment between each update
BATCH_SIZE = n_env*n_step       # Data collected for each update
MAX_EPOCH = 30

# Hyperparameters
LR = 2.5e-4                     # Optimizer learning rate
GAMMA = 0.9                     # Discount factor
GAE_LAMBDA = 0.99               # TD(lambda) factor: 1 -> 
K_EPOCHS = 4                    # Number of update at the end data collection

CLIP = 0.2                      # Clipping factor in policy loss
ENTROPY_COEF = 0.01             # Entropy coefficent for loss calculation
VALUE_COEFF = 0.5               # Value coefficent for loss calculation

MINI_BATCH_SIZE = BATCH_SIZE//K_EPOCHS      # Be careful here

# Test parameters
TEST_INTERVAL = 10
RECORD_VIDEO = 50

"""
-----------------------------------------------------------------------
                            Utils function
-----------------------------------------------------------------------
"""

# Tensorboard log creation
from torch.utils.tensorboard import SummaryWriter
from md_report import create_md_summary
from collections import deque

class Logger:
    def __init__(self, gym_id: str,name: str) -> SummaryWriter:
        
        if name is not None:

            # Create tensorboard logger
            self.logger = SummaryWriter("debug/" + name)
            # Create a md file for hyperparam
            create_md_summary(gym_id, name)

        else:
            self.logger = None
        print(f"Experiment name: {name}")

        self.loss_index = 0
        self.train_index = 0
        self.test_index = 0

        self.timer = time()
        self.buff = deque(maxlen=100)

    def add_train_rew(self, reward: int, tag: str = "Train/Episode Reward") -> None:

        if self.logger is not None:
            self.logger.add_scalar(tag, reward, self.loss_index)
            self.loss_index += 1
        
    def add_loss(self, loss: int, tag: str = "Train/Loss") -> None:
        
        if self.logger is not None:
            if type(loss) == float:
                self.logger.add_scalar(tag, loss, self.loss_index)
            else: 
                self.logger.add_scalar(tag, loss.item(), self.loss_index)
            self.loss_index += 1

    def add_test(self, reward: int, tag: str = "Test/Reward") -> None:

        if self.logger is not None:
            self.logger.add_scalar(tag, reward, self.test_index)
            self.test_index +=1
    
    def close(self):
        if self.logger is not None:
            self.logger.flush()
            self.logger.close()

    def show_progress(self, update) -> None: 
        
        if update != 1:
            dt = time()-self.timer
            epoch_speed = 1/dt
        else: 
            epoch_speed = 0
        self.timer = time()
        self.buff.append(epoch_speed)
        avg = sum(self.buff)/len(self.buff)
        print(f"\rProgress: {update/MAX_EPOCH*100:2.2f} % \t Epoch/s: {epoch_speed:.2f} \t Average speed: {avg:.2f}", end="")


# Function for making vector enviroment
import gymnasium as gym
import torch
import numpy as np
import random
from time import time

def make_env(gym_id: str,idx: int, rnd: bool = False) -> gym.spaces:
    def alias():

        # Change env name here
        env = gym.make(gym_id)
        
        # Set a random seed if specified
        if rnd:
            seed = time()
        else:
            seed = 92 
        """
        env.seed(seed+idx)
        env.action_space.seed(seed+idx)
        env.observation_space.seed(seed+idx)
        """
        return env
    
    return alias

def set_seed(rnd: bool = False) -> None:
    """
    Function to set seed on all packages exept for gymansium
    :param rnd: Flag for random seed, if true use time() as seed
    """
    if rnd:
        random.seed(time())
        np.random.seed(time())
        torch.manual_seed(time())
        torch.backends.cudnn.deterministic = False
        print("Using Random seed")
    else:
        random.seed(92)
        np.random.seed(92)
        torch.manual_seed(92)
        torch.backends.cudnn.deterministic = True
        print("Using deterministic seed")

def test_netwrok(update, agent, test_env, logger):
    """
    Execute a complete run in a test enviroment without exploration
    """
    if update % TEST_INTERVAL:
        stop_test = False
        test_reward = 0
        test_state, _ = test_env.reset()
        
        while not stop_test:
            # Get action with argmax
            with torch.no_grad():
                test_state_tensor = torch.tensor(test_state)
                test_action = agent.get_action_test(test_state_tensor)

            ns, rw, ter, trun, _ = test_env.step(test_action.numpy())
            test_reward += rw
            test_state = ns

            if ter or trun:
                logger.add_test(test_reward)
                stop_test = True