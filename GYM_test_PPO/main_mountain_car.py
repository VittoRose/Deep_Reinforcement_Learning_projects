# Tring to solve Mountain car gym envoroment by modifing the reward function

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCritic import Agent
from parameters import *

# Aggiungere seed, cuda, misura della velocità

def state2reward(state, train) -> int:
    """
    Modified reward function, reward are now proportional to x position
    """
    if train:
        reward = torch.zeros(len(state))
        for i in range(len(state)):
            x = state[i][1]

            rw = 10*abs(x)

            reward[i] = rw
    else:
        x = state[1]
        rw = 10*abs(x)
        reward = rw

    return reward

if __name__ == "__main__":

    # Name the experiment
    name = "mountain_car_new_reward1"
    gym_id = "MountainCar-v0"

    # Tensorboard Summary writer
    logger = make_logger(gym_id, name)

    # Experiment run with deterministic seed, to use random seed modify also enviroment
    set_seed(False)

    # Vector enviroment object, change rnd to true for random seed
    envs = gym.vector.SyncVectorEnv([make_env(gym_id,i, rnd=False) for i in range(n_env)])
    
    # Test enviroment
    test_env = gym.make(gym_id, render_mode = "rgb_array")
    test_env = gym.wrappers.RecordVideo(test_env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: x % 100 == 0)

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
    rw_place = 0
    ls_place = 0
    test_place = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0             # Number of enviroment step
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs)
    next_done = torch.zeros(n_env)
    num_updates = MAX_ITERATION // BATCH_SIZE
    test_counter = 0

    """
    ------------------------------------------------------------
                        TRAINING LOOP
    ------------------------------------------------------------
    """

    for update in range(1, num_updates + 1):

        # Show progress during training
        print(f"\r Progress: {update/num_updates*100:2.2f} %", end="")

        # Here we can modify the learning rate

        # Collect data from the enviroment
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

            # Execute action in enviroment
            next_obs, _, truncated, terminated, info = envs.step(action.numpy())

            # Here we get the new reward function
            rewards[step] = state2reward(next_obs, train=True)

            done = terminated | truncated
            next_obs, next_done = torch.tensor(next_obs), torch.tensor(done)

            # Collect rewards per episode
            with torch.no_grad():
                ep_reward = ep_reward + rewards[step]

            for i in range(len(done)):
                if done[i]:
                    if logger is not None:
                        logger.add_scalar("Train/Episode Reward", ep_reward[i], rw_place)
                        rw_place += 1
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

            # Shuffle index to break correlations
            np.random.shuffle(index)      

            # Update using minibatches
            for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):     
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

                if logger is not None:
                    logger.add_scalar("Train/Loss", loss.item(), ls_place)
                    ls_place +=1

                optimizer.zero_grad()
                loss.backward()
                # TODO: be sure about clipping gradient norm
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                test_counter += 1

            if test_counter % TEST_INTERVAL == 0:
                stop_test = False
                test_reward = 0
                test_state, _ = test_env.reset()

                while not stop_test:
                    # Get action with argmax
                    with torch.no_grad():
                        test_state_tensor = torch.tensor(test_state)
                        test_action = agent.get_action_test(test_state_tensor)

                    ns, _, ter, trun, _ = test_env.step(test_action.numpy())

                    # New reward function
                    rw = state2reward(ns, train=False)

                    test_reward += rw
                    test_state = ns

                    if ter or trun:
                        if logger is not None:
                            logger.add_scalar("Test/Reward", test_reward, test_place)
                            test_reward = 0
                            test_place += 1
                        stop_test = True

    # Close api
    test_env.close()                
    envs.close()

    if logger is not None:
        logger.close()

    print("\nTraining over") 