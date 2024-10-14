import torch
from parameters import *
from copy import deepcopy

def get_action(actor, state) -> tuple[torch.tensor, torch.tensor]:
    
    with torch.no_grad():
        action_probs = actor(state)

        action = torch.multinomial(action_probs, 1)

        prob_action = action_probs[action].item()

    #print("Network output: ", action_probs, "Action choosen: ", action, " with probability: ", prob_action)

    return action, prob_action

def calculate_advantage(rewards, terminated, values, value_end, T, gamma:int = GAMMA, lam:int = LAMBDA):
     
    # Preallocation
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(1)
    value_target = torch.zeros_like(rewards)

    for t in reversed(range(T)):
        if t == T-1:
            next_value = value_end
        else:
            next_value = values[t]
        
        # TD error: δ_t = r_t + (1 - done) *γ * V(s_{t+1})  - V(s_t)
        delta_t = rewards[t] + (1-terminated[t])*gamma*next_value - values[t]

        # Advantage for actor update
        gae = delta_t + (1-terminated[t])*gamma*lam*gae
        advantages[t] = gae 

        # Value target for critic update
        value_target[t] = advantages[t] + values[t]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


    return advantages, value_target

def calculate_advantage_new(rewards, terminated, values, value_end, T:int, gamma:int = GAMMA):
    # Preallocation
    advanteges = torch.zeros_like(rewards)
    reward = torch.zeros_like(rewards)
    discounted_rew = torch.zeros(1)
    value_target = torch.zeros_like(rewards)

    for t in reversed(range(T)):
        if t == T-1:
            next_value = value_end
        else:
            next_value = values[t]
        
        discounted_rew = (1-terminated[t])*(rewards[t] + gamma*discounted_rew)
        
        advanteges[t] = discounted_rew - next_value

        # Value target for critic update
        value_target[t] = discounted_rew


    return advanteges, value_target

def PPO_loss(advantages, prob_action, old_prob_action) -> torch.tensor:
    
    # Finding the ratio (pi_theta / pi_theta__old)
    ratio = prob_action/old_prob_action

    surr1 = ratio*advantages
    surr2 = torch.clamp(ratio, 1-CLIP, 1+CLIP)*advantages

    losses = torch.min(surr1, surr2)

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

    return loss


class old_network():
    def __init__(self, network):
        self.network = deepcopy(network)

    def get_prob(self, state, action):

            action_prob = self.network(state)

            prob_action = action_prob[action].item()

            #print("OLD Network output: ", prob_action)

            return prob_action
    
    def update(self, network):
         self.network.load_state_dict(network.state_dict())