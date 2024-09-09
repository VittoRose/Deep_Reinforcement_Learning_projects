import Data_struct
import DNQ
import Enviroment
import Graphics
import torch
from parameters import *
from torch.utils.tensorboard import SummaryWriter
from time import time


# Enviroment param
n_env = 5
epidosde = 8
state_size = 4
action_size = 5


# Create a writer to save data during training
logger = SummaryWriter(f"logs/Buffer10k_Batch50")

# Create Neural network
RL_agent = DNQ.Agent(state_size, action_size)

# Create replay buffer
buffer = Data_struct.DictBuffer(state_size, n_envs=n_env)

# Grapichs management
gui = Graphics.GUI()
gui.GUI_init(time())


test_env = Enviroment.PendolumEnv()

# Create a policy
policy = DNQ.Policy(RL_agent, torch.optim.Adam(RL_agent.parameters(), lr=LR), buffer, DISCOUNT_FACTOR, EPS, logger, UPDATE_INTERVAL )

# Pool of train events
train_env = Enviroment.VectorEnv([Enviroment.PendolumEnv() for _ in range(n_env)])

# Collector for training enviroment
train_collector = Data_struct.Collector(policy, train_env, buffer, action_size, logger)

# Fill the buffer with data
train_collector.collect(n_exp=buffer.capacity)

# Loop variable
running = True
reset, i, ep_reward = 0, 0, 0

# Initial test condition
state = test_env.reset()
ang = test_env.ang


# ---------------------------------------------------- #
#                   Training loop                      #
# ---------------------------------------------------- #

try:
    while running:

        # Check for termination condition
        running = gui.GUI_quit()

        # Collect some experience in the buffer with exploration
        train_collector.collect(n_exp = 2*BATCH_SIZE)
        
        # Perform a learn step
        policy.learn()

        # Test target network


        # Get q_value from the target network
        q_test = policy.target(state)

        # Select the action
        action = torch.argmax(q_test)

        # Perform a step in the enviroment
        state, reward, terminated, truncated =  test_env.step(action)
        ep_reward += reward

        # Update the angle 
        ang = test_env.ang

        # Reset enviroment if the 
        if terminated or truncated:
            state = test_env.reset()

            if logger is not None:
                logger.add_scalar("Test/reward", ep_reward, reset)

            ep_reward = 0
            reset += 1

        
        gui.draw(ang, reset, ep_reward)

        i+=1

        if i % 1000 == 0:
            print(f"Number of learning steps: {i}")

except KeyboardInterrupt:
    running = False

if logger is not None:
    logger.flush()
    logger.close()

print("Script ended successfully")