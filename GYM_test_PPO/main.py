import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ActorCritic import Agent
from parameters import *
from utils.run_info import InfoPlot
from utils.util_function import make_env, test_netwrok


# Name the experiment
name = "acrobot_04"
gym_id = "Acrobot-v1"

# Tensorboard Summary writer
logger = InfoPlot(gym_id, name, "logs_acrobot/")

# Vector enviroment object, change rnd to true for random seed
envs = gym.vector.SyncVectorEnv([make_env(gym_id,i, rnd=True) for i in range(n_env)])

# Test enviroment
test_env = gym.make(gym_id, render_mode="rgb_array")

"""
if name is not None:
    test_env = gym.wrappers.RecordVideo(test_env,
                                        f"videos/{name}",
                                        episode_trigger=lambda x: x % RECORD_VIDEO == 0)
"""


# RL agent and optimizer
agent = Agent(envs)
optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

# Buffer preallocation
obs = torch.zeros((n_step, n_env) + envs.single_observation_space.shape)
actions = torch.zeros((n_step, n_env) + envs.single_action_space.shape)
logprobs = torch.zeros((n_step, n_env))
rewards = torch.zeros((n_step, n_env))
dones = torch.zeros((n_step, n_env))
values = torch.zeros((n_step, n_env))

# Collect reward to plot
ep_reward = torch.tensor(n_env)

next_obs, _ = envs.reset()
next_obs = torch.tensor(next_obs)
next_done = torch.zeros(n_env)

"""
------------------------------------------------------------
                    TRAINING LOOP
------------------------------------------------------------
"""

for epoch in range(0, MAX_EPOCH):

    # Show progress during training
    logger.show_progress(epoch)

    test_netwrok(epoch, agent, test_env, logger)
    
    # Here we can modify the learning rate

    # Collect data from the enviroment
    for step in range(0, n_step):

        obs[step] = next_obs
        dones[step] = next_done

        # Select an action
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # Execute action in enviroment
        next_obs, reward, truncated, terminated, _ = envs.step(action.numpy())
        done = terminated | truncated

        rewards[step] = torch.tensor(reward)
        next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)

        # Collect rewards per episode
        with torch.no_grad():
            ep_reward = ep_reward + rewards[step]

        for i, cut in enumerate(done):
            if cut:
                logger.add_train_rew(ep_reward[i])
                ep_reward[i] = 0

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(n_step)):
            if t == n_step- 1:
                nextnonterminal = 1.0 - next_done.int()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1].int()
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    index = np.arange(BATCH_SIZE)

    # Update network K times
    for epoch in range(K_EPOCHS):

        # Shuffle index to break correlations
        np.random.shuffle(index)

        # Update using minibatches
        for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):     
            
            # Select the index for minibatch
            end = start + MINI_BATCH_SIZE
            mini_batch_index = index[start:end]

            _, newlogprob, entropy, newval = agent.get_action_and_value(b_obs[mini_batch_index], b_actions.long()[mini_batch_index])
            logratio = newlogprob - b_logprobs[mini_batch_index]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mini_batch_index]

            # Normalize advantages
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Policy loss
            surr1 = -mb_advantages * ratio
            surr2 = -mb_advantages * torch.clamp(ratio, 1-CLIP, 1+CLIP)
            pg_loss = torch.max(surr1, surr2).mean()

            # Value loss
            # TODO: add clipped value loss
            v_losses = torch.nn.functional.mse_loss(newval.squeeze(), b_returns[mini_batch_index])
            v_loss = v_losses.mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            # Global loss function
            loss = pg_loss - ENTROPY_COEF*entropy_loss + VALUE_COEFF*v_loss
            logger.add_loss(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


# Close api
test_env.close()                
envs.close()
logger.close()

print("\nTraining over") 