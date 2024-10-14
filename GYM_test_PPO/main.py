import torch
import gymnasium as gym
import Network
from parameters import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
from time import time

# Enviroment
#env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = env.reset()

# Network
actor = Network.Actor(len(state), 2)
critic = Network.Critic(len(state))
old_actor = old_network(actor) 

# Optimizer
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

name = "minibatch"

logger = SummaryWriter("logs/" + name)

running = True

rewards = torch.zeros(T)
values = torch.zeros(T)
done_term = torch.zeros(T)
done_trunc = torch.zeros(T)

prob_actions = torch.zeros(T)
prob_actions_old = torch.zeros(T)

iteration = 0
ep_reward = 0
reset = 0
print("")

# Save starting time
t0 = time()

try:
    while running and iteration < MAX_ITERATION:
        
        # Perform T step in the enviroment and store transition data
        for t in range(T):

            # Get an action from actor
            action, prob_actions[t] = get_action(actor, state)

            # Evaluate the prob to chose the same action with the old network
            prob_actions_old[t] = old_actor.get_prob(state, action)

            # Evaluate the value function with the critic network
            value = critic(state)

            # Perform step in the enviromen and store all data
            next_state, reward, terminated, truncated, _ = env.step(int(action))

            ep_reward += reward
            
            rewards[t] = reward
            values[t] = value
            done_term[t] = terminated 
            done_trunc[t] = truncated

            # Last step -> eval V(s_{t+1})
            if t == T-1:
                next_value = critic(next_state)

            # Transition for next iteration
            if terminated or truncated: 
                state, _ = env.reset()

                logger.add_scalar("Rewards", ep_reward, reset)
                print(f"\rEpisode reward: {ep_reward}\t Progress: {iteration/MAX_ITERATION*100+0.01:.2f} %", end="")
                reset += 1
                ep_reward = 0

            else: 
                state = next_state

        # Update networks using minibatches

        for i in range(K):

            index = (i*MINI_BATCH, (i+1)*MINI_BATCH)

            # Compute quantities to update NN
            advantage, value_target = calculate_advantage(rewards, done_term, values, next_value, index)

            # Compute loss for actor
            loss = PPO_loss(advantage, prob_actions, prob_actions_old, index)

            logger.add_scalar("Loss actor", loss.item(), iteration)

            actor_update(loss, actor, actor_optim, old_actor)

            loss_c = critic_update(values, value_target, critic_optim, index)
            logger.add_scalar("Loss critic", loss_c, iteration)

            if i>2:
                print("asdasda")
                quit()

        # Reset buffer
        values = torch.zeros(T)
        rewards = torch.zeros(T)
        done_term = torch.zeros(T)
        done_trunc = torch.zeros(T)

        prob_actions = torch.zeros(T)
        prob_actions_old = torch.zeros(T)

        iteration +=1


except KeyboardInterrupt:
    running = False

print("\nTraining done")
print(f"Time: {(time()- t0)/60:.2} min")
# ----------------------------------
#           TEST NETWORK
# ----------------------------------

state, _ = env.reset()
ep_len = 0
ep_rw = 0
reset = 0

try:
    
    while reset <= TEST_RESET and running:

        with torch.nograd():    
            action_prbs = actor(state)
            action = torch.argmax(action_prbs)

        ns, rw, ter, trun, _ = env.step(action)

        ep_rw += rw
        if ter or trun:
            logger("Test/Episode reward: ", ep_rw, reset)
            logger("Test/Episode len: ", ep_len, reset)
            
            # Update counting variable
            reset += 1
            ep_len = 0
            ep_rw = 0
        else:
            ep_len +=1   
        

except KeyboardInterrupt:
    running = False    

env.close()

logger.flush()
logger.close()

actor.save("NN/" + name)
