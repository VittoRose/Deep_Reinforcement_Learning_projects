import torch
from parameters import *
import numpy as np
from copy import deepcopy

def get_action(actor, state) -> tuple[torch.tensor, torch.tensor]:
    
    with torch.no_grad():
        action_probs = actor(state)

        distr = torch.distributions.Categorical(action_probs)

        action = distr.sample()

        log_prob = distr.log_prob(action)

    #print("Network output: ", action_probs, "Action choosen: ", action, " with probability: ", prob_action)

    return action, log_prob

def calculate_advantage(rewards, terminated, values, value_end, T, gamma:int = GAMMA, lam:int = LAMBDA):
     
    # Preallocation
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(1)
    value_target = torch.zeros_like(rewards)
    discounted_rew = torch.zeros(1)

    for t in reversed(range(T)):
        if t == T-1:
            next_value = value_end
        else:
            next_value = values[t]
        
        # TD error: δ_t = r_t + (1 - done) *γ * V(s_{t+1}) - V(s_t)
        delta_t = rewards[t] + (1-terminated[t])*gamma*next_value - values[t]

        # Advantage for actor update
        gae = delta_t + (1-terminated[t])*gamma*lam*gae
        advantages[t] = gae

        # Value target for critic update
        discounted_rew = (1-terminated[t])*(rewards[t] + gamma*discounted_rew) 
        value_target[t] = discounted_rew

    # Normalize advantages, not teoretically necessary but suggested 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, value_target

def calculate_advantage_new(rewards, terminated, values, value_end, T:int, gamma:int = GAMMA):
    # Preallocation
    rew_list = []
    discounted_rew = 0

    for t in reversed(range(T)):
      
        if terminated[t]:
            discounted_rew = 0
        discounted_rew = rewards[t] + gamma*discounted_rew
        rew_list.insert(0, discounted_rew)

    # Normalize reward vector
    rew_list = torch.tensor(rew_list, dtype=torch.float32)
    rew_list = (rew_list-rew_list.mean())/(rew_list.std() + 1e-7)

    advanteges = rew_list.detach() - values

    return advanteges, rew_list

def PPO_loss(advantages, prob_action, old_prob_action, value, rewards) -> torch.tensor:
    
    # Finding the ratio (pi_theta / pi_theta__old) with exp(log(pi) - log(pi_old))
    ratio = torch.exp(prob_action - old_prob_action)

    surr1 = ratio*advantages
    surr2 = torch.clamp(ratio, 1-CLIP, 1+CLIP)*advantages

    # print("surr1 dim: ", surr1.shape, "surr2 dim: ", surr2.shape, "value dim: ", value.shape, "rew dim: ", rewards.shape)

    losses = -torch.min(surr1, surr2) 

    # Mean loss across enviroment and timesteps   
    loss = -losses.mean()

    #print("Loss: ", loss)

    return loss

def actor_update(loss, actor, actor_optimizer, old_actor) -> None:
    
    # Update old network before backward pass
    old_actor.update(actor)

    # Backpropagation
    actor_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    actor_optimizer.step()

def critic_update(values, value_target, crit_optim) -> int:

    # Evaluate MSE loss
    loss = torch.nn.functional.mse_loss(values, value_target)

    out = loss.item()

    # Backpropagation
    crit_optim.zero_grad()
    loss.backward(retain_graph=True)
    crit_optim.step()

    return out

def update_prob_action(actor, state_buffer, action_buffer) -> torch.tensor:
    # Recalculate log_prob for each state in the buffer with the updated weight
    new_prob = torch.zeros(T)

    for i in range(len(state_buffer)):
        action_prob = actor(np.array(state_buffer[i]))
        dist = torch.distributions.Categorical(action_prob)
        action_log_prob = dist.log_prob(torch.tensor(action_buffer[i]))
        new_prob[i] = action_log_prob

    return new_prob
        

class old_network():
    def __init__(self, network):
        self.network = deepcopy(network)

    def get_prob(self, state, action):
            """
            action_prob = self.network(state)

            prob_action = action_prob[action].item()
            """

            with torch.no_grad():
                action_probs = self.network(state)

                distr = torch.distributions.Categorical(action_probs)

                log_prob = distr.log_prob(action)

            #print("OLD Network output: ", log_prob)
            return log_prob
    
    def update(self, network):
         self.network.load_state_dict(network.state_dict())