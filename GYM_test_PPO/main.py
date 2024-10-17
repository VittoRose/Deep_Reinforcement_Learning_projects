# Working with parallel enviroment

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCritic import Agent
from parameters import *

# Tensorboard Summary writer
logger = make_logger(None)

def make_env():
    return gym.make("CartPole-v1")

if __name__ == "__main__":

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(n_env)])
    agent = Agent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # Buffer preallocation
    obs = torch.zeros((n_step, n_env) + envs.single_observation_space.shape)
    actions = torch.zeros((n_step, n_env) + envs.single_action_space.shape)
    logprobs = torch.zeros((n_step, n_env))
    rewards = torch.zeros((n_step, n_env))
    dones = torch.zeros((n_step, n_env))
    values = torch.zeros((n_step, n_env))

    ep_reward = torch.tensor(n_env)
    place = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0             # Number of enviroment step
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs)
    next_done = torch.zeros(n_env)
    num_updates = MAX_ITERATION // BATCH_SIZE

    for update in range(1, num_updates + 1):

        # Here we can modify the learning rate

        for step in range(0, n_step):
            global_step += 1*n_env
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, truncated, terminated, info = envs.step(action.numpy())
            done = terminated | truncated

            rewards[step] = torch.tensor(reward)
            next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)

            with torch.no_grad():
                ep_reward = ep_reward + rewards[step]

            for i in range(len(done)):
                if done[i]:
                    if logger is not None:
                        logger.add_scalar("Train/Episode Reward", ep_reward[i], place)
                        place += 1
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

        # Optimizing the policy and value network
        index = np.arange(BATCH_SIZE)
        clipfracs = []

        # Update network K times
        for epoch in range(K_EPOCHS):
            np.random.shuffle(index)      # Shuffle index, idk why, io lo toglierei
            for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):     # from 0 to batch_size, minibatch step
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

                # TODO: entropy loss and v_loss has an hyperparam
                loss = pg_loss - ENTROPY_COEF*entropy_loss + VALUE_COEFF*v_loss

                optimizer.zero_grad()
                loss.backward()
                # TODO: be sure about clipping gradient norm
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    envs.close()
    if logger is not None:
        logger.close()

    print("Training over")

