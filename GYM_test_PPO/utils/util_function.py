# Function for making vector enviroment
import gymnasium as gym
import torch
from parameters import *
from time import time

def make_env(gym_id: str, idx: int, rnd: bool = False) -> gym.spaces:
    def alias():
        # If enviroment need more args add them here
        env = gym.make(gym_id)
        
        # Set seed     
        if not rnd: 
            env.np_random = SEED + idx
        else:
            env.np_random = time() + idx
        return env
    return alias

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