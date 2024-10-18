
# Training parameters
n_env = 4
n_step = 128                    # Number of step in the enviroment between each update
BATCH_SIZE = n_env*n_step       # Data collected for each update
MAX_ITERATION = 1_000_000

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
def make_logger(gym_id: str,name: str) -> SummaryWriter:
    if name is not None:

        # Create tensorboard logger
        logger = SummaryWriter("logs/" + name)
        
        # Create a md file for hyperparam
        create_md_summary(gym_id, name)

    else:
        logger = None
    print(f"Experiment name: {name}")

    return logger

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