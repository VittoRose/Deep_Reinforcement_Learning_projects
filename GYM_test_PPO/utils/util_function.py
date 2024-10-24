# Function for making vector enviroment
import gymnasium as gym
import torch
from parameters import *
from time import time
import numpy as np

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

def test_network(update, agent, test_env, logger):
    """
    Execute 5 complete run in a test enviroment without exploration
    """
    if update % TEST_INTERVAL:
        
        rew_data = np.zeros(5)
        len_data = np.zeros(5)
        
        # Collect data for 5 episode of test and log the mean reward and ep_lenght
        for i in range(5):
            stop_test = False
            test_reward = 0
            test_state, _ = test_env.reset()
            ep_len = 0
            
            while not stop_test:
                # Get action with argmax
                with torch.no_grad():
                    test_state_tensor = torch.tensor(test_state)
                    test_action = agent.get_action_test(test_state_tensor)

                ns, rw, ter, trun, _ = test_env.step(test_action.numpy())
                test_reward += rw
                test_state = ns
                ep_len +=1

                if ter or trun:
                    rew_data[i] = test_reward
                    len_data[i] = ep_len
                    stop_test = True



        if ter or trun:
            logger.add_test(np.mean(rew_data), np.mean(ep_len))
            stop_test = True