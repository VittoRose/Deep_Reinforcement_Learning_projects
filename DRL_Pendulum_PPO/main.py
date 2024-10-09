import Buffer
import PPO
import Enviroment
import Collector 
import torch
from parameters import *
import Graphics
import Network
from torch.utils.tensorboard import SummaryWriter
from time import time

# Enviroment param
n_env = 5
state_size = 4
action_size = 5

# Create a writer to save data during training
logger = SummaryWriter(f"logs/cancella")
#logger = None

# Create Neural network
actor = Network.Actor(state_size, action_size)
critic = Network.Critic(state_size)

# Pack NN
networks = (actor, critic)

# Optimizer for the two network
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR, eps=1e-5)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC, eps=1e-5)

# Pack optimizer
optimizers = (actor_optimizer, critic_optimizer)

# Create replay buffer
buffer = Buffer.Buffer(actor, n_envs=n_env)

# Create a policy
trainer = PPO.Algorithm(networks, optimizers, buffer, GAMMA, CLIP, LAMBDA, n_env, logger)

# Pool of train events
train_env = Enviroment.VectorEnv([Enviroment.PendolumEnv() for _ in range(n_env)])

# Collector for training enviroment
train_collector = Collector.Collector(trainer, train_env, buffer, action_size, logger)

# Grapichs management
#gui = Graphics.GUI()
#gui.GUI_init(time())

# Test enviroment
#test_env = Enviroment.PendolumEnv()

# Loop variable
running = True
reset, i, ep_reward = 0, 0, 0

# Initial test condition
#state = test_env.reset()
#ang = test_env.ang

t0 = time()

# ---------------------------------------------------- #
#                   Training loop                      #
# ---------------------------------------------------- #

try:
    while running:

        # Check for termination condition
        #running = gui.GUI_quit()

        # Collect BUFFER_SIZE timestep of experience from each enviroment 
        train_collector.collect(BUFFER_SIZE)

        # Evaluate advantages approximation
        advantages, value_target = trainer.calculate_advantage()

        trainer.actor_update(advantages, i)

        trainer.critic_update(value_target)

        # --------------------
        # Test target network
        # --------------------
        """
        # Get q_value from the target network
        with torch.no_grad():
            q_test = trainer.actor(state)

        # Select the action
        action = torch.argmax(q_test)

        # Perform a step in the enviroment
        state, reward, terminated, truncated = test_env.step(action)
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

        """
        i+=1

        if i % 1000 == 0:
            print(f"Number of learning steps: {i}, time elapsed {time()-t0}")
            t0 = time()

except KeyboardInterrupt:
    running = False

if logger is not None:
    logger.flush()
    logger.close()

actor.save("NN/Test0")

print("Script ended successfully")