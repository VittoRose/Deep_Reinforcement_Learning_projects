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

obs_dim = len(state)
act_dim = 2

# Network
actor = Network.Actor(obs_dim, act_dim)
critic = Network.Critic(obs_dim)
old_actor = old_network(actor) 

# Optimizer
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR, eps=1e-5)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC, eps=1e-5)

# Tensorboard logger
name = "different_losses"
if name is not None:
    logger = SummaryWriter("logs/" + name)
else:
    logger = None
print(f"Tensorboard logger: {name}")

# Preallocation for buffer
rewards = torch.zeros(T)
values = torch.zeros(T)
done_term = torch.zeros(T)
done_trunc = torch.zeros(T)
prob_actions = torch.zeros(T)                        # log(probs)
prob_actions_old = torch.zeros(T)
state_buffer = []
action_buffer = []

# Loop variable
iteration = 0
ep_reward = 0
reset = 0
running = True

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
            state_buffer.append(state)
            action_buffer.append(int(action))

            # Last step -> eval V(s_{t+1})
            if t == T-1:
                next_value = critic(next_state)

            # Transition for next iteration
            if terminated or truncated: 
                state, _ = env.reset()
                
                if logger is not None:
                    logger.add_scalar("Rewards", ep_reward, reset)
                
                print(f"\rEpisode reward: {ep_reward}\t Progress: {iteration/MAX_ITERATION*100+0.01:.2f} %", end="")
                reset += 1
                ep_reward = 0

            else: 
                state = next_state
            

        #for ii in range(K):
            
        # Compute quantities to update NN
        advantage, value_target = calculate_advantage_new(rewards, done_term, values, next_value, T)

        # Compute loss for actor
        loss = PPO_loss(advantage, prob_actions, prob_actions_old, value_target, values)

        actor_update(loss, actor, actor_optim, old_actor)

        loss_c = critic_update(values, value_target, critic_optim)

        if logger is not None:
            logger.add_scalar("Loss actor", loss.item(), iteration)
            logger.add_scalar("Loss critic", loss_c, iteration)

        """
            # Recalculate values, prob actions with the new network
            if ii != K-1:
                prob_actions_old = prob_actions
                values = critic(np.array(state_buffer)).squeeze()
                prob_actions = update_prob_action(actor, state_buffer, action_buffer)
            
            print("Prob action ", prob_actions.shape , ii)
            print("Prob action old ", prob_actions_old.shape, ii)
            print("Values dim: ", values.shape)
        """

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
print(f"Time: {(time()- t0)/60:.0f} min {(time()-t0)%60:.0f} s")
# ----------------------------------
#           TEST NETWORK
# ----------------------------------

state, _ = env.reset()
ep_len = 0
ep_rw = 0
reset = 0

# Save test starting time
t1 = time()
try:
    
    while reset <= TEST_RESET and running:

        # Select the action with more probability
        with torch.no_grad():    
            action_prbs = actor(state)
            action = torch.argmax(action_prbs)

        # Get data from the enviroment
        ns, rw, ter, trun, _ = env.step(int(action))

        ep_rw += rw
        if ter or trun:
            if logger is not None:
                logger.add_scalar("Test/Episode reward: ", ep_rw, reset)
                logger.add_scalar("Test/Episode len: ", ep_len, reset)
            
            state, _ = env.reset()

            # Update counting variable
            reset += 1
            ep_len = 0
            ep_rw = 0
        else:
            ep_len +=1
            state = ns   
        

except KeyboardInterrupt:
    running = False    

env.close()

if logger is not None:
    logger.flush()
    logger.close()

if name is not None:
    actor.save("NN/" + name)

print(f"Test over, time: {(time()-t1)/60:.0f} min {(time()-t1)%60:.0f} s")