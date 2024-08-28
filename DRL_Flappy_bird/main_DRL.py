import DNQ
import Enviroment
from parameters import *
import torch
import Jumping_bird as jb
from torch.utils.tensorboard import SummaryWriter
from time import time
import Data_struct
import random

# Create a writer to save data during training
#logger = SummaryWriter("logs/Buffer100k_Batch50")
logger = None

# Get a random seed
random.seed(time())

# Create a test enviroment
test_env = Enviroment.FlappyBird()

# Create a object to display test data
screen = jb.Graphics()

# Create a replaybuffer
buffer = Data_struct.DictBuffer(test_env.state_size, n_envs=N_ENV)

# Create Neural network
RL_agent = DNQ.Agent(test_env.state_size, test_env.action_size)

# Create a policy
policy = DNQ.Policy(RL_agent, torch.optim.Adam(RL_agent.parameters(), lr=LR), buffer, DISCOUNT_FACTOR, EPS, logger, UPDATE_INTERVAL )

# Pool of train events
train_env = Enviroment.VectorEnv([Enviroment.FlappyBird() for _ in range(N_ENV)])

# Collector for training enviroment
train_collector = Data_struct.Collector(policy, train_env, buffer, test_env.action_size, logger)

# Fill the buffer with data
train_collector.collect(n_exp=buffer.capacity)

# Loop variable
running = True
reset, ep_lenght, ep_reward = 0, 0, 0

# Initial test condition
state = test_env.reset()

# Save training loop start time
t0 = time()

# Loop counter variable
counter = 0

# ---------------------------------------------------- #
#                   Training loop                      #
# ---------------------------------------------------- #

try:
    while running:

        # Check for termination condition
        running, _ = screen.user_interaction()


        # Collect some experience in the buffer with exploration
        train_collector.collect(n_exp = 2*BATCH_SIZE)
        
        # Perform a learn step
        policy.learn()

        # Test target network
        # ------------------ #

        # Get q_value from the target network
        q_test = policy.target(state)

        # Select the action
        action = torch.argmax(q_test)

        # Perform a step in the enviroment
        state, reward, terminated, truncated =  test_env.step(action)
        
        # Logger variable
        ep_reward += reward
        ep_lenght += 1

        # Reset enviroment if the 
        if terminated or truncated:
            state = test_env.reset()

            if logger is not None:
                logger.add_scalar("Test/Reward", ep_reward, reset)
                logger.add_scalar("Test/Length", ep_lenght, reset)
            
            # Reset episode stats
            ep_reward = 0
            ep_lenght = 0

            reset += 1

        bird, pipes = test_env.get_disp_data()

        screen.draw(bird, pipes, reset, test_env.score, ep_reward)

        counter += 1

        if counter % PLOT_LEARNING_STEP == 0:
            print(f"Learning step: {counter} \t Time from last print: {time()-t0}")
            t0 = time()     # Count time from here

except KeyboardInterrupt:
    running = False

RL_agent.save()

if logger is not None:
    logger.flush()
    logger.close()

print("Script ended successfully")