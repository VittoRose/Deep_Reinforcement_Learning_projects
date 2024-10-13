import torch
from torch import nn
import numpy as np
from copy import deepcopy
from parameters import *
    
class Algorithm():

    def __init__(self, networks, optimizers, buffer, discount_factor, epsilon, attenuation, n_env, writer = None):
        
        # Store NN
        self.actor = networks[0]
        self.critic = networks[1]

        # Copy actor network for rateo evaluation
        self.old_actor = deepcopy(self.actor)

        # Store optimizer
        self.a_optim = optimizers[0]
        self.c_optim = optimizers[1]

        # Store buffer
        self.buffer = buffer

        # Tensorboard logger
        self.writer = writer

        # Discount factor
        self.gamma = discount_factor
        
        # Clip param
        self.eps = epsilon

        # Number of parallels enviroment
        self.n_env = n_env

        # Lambda param
        self.lam = attenuation

    def ppo_loss(self, advantages) -> torch.tensor:
        """
        Loss function for PPO algorithm
        """
        
        prob = self.buffer.prob
        old_prob = self.buffer.old_prob

        """
        rateo = torch.zeros_like(prob)
        losses = torch.zeros_like(prob)

        # Evaluate loss for each enviroment and timestamp
        for t in range(0,len(prob)):

            rateo[:,t] = prob[:,t]/old_prob[:,t]
            
            no_clip = rateo[:,t]*advantages[:,t]
            clip = torch.clamp(rateo[:,t], 1-CLIP, 1+CLIP)*advantages[:,t]

            losses[:,t] = -torch.min(clip, no_clip)
        """

        # Evaluate loss for each enviroment and timestamp
        rateo = prob/old_prob
        
        no_clip = rateo*advantages
        clip = torch.clamp(rateo, 1-CLIP, 1+CLIP)*advantages

        losses = -torch.min(no_clip, clip)

        # Mean loss across enviroment and timesteps   
        loss = losses.mean()

        return loss


    def calculate_advantage(self):
        """
        Generalized Advantage Estimation (GAE) for parallel enviroments
        """

        # Get data from buffer
        rewards, dones, values = self.buffer.get_transition()        

        # Init advantages vector
        advantages = torch.zeros_like(rewards)
        value_target = torch.zeros_like(rewards)
        gae = torch.zeros(self.n_env)
        
        for t in reversed(range(self.buffer.capacity)):

            if t == self.buffer.capacity - 1:                   # Last timestep 
                next_non_terminal = 1.0 - dones[:, -1]  
                next_value = self.buffer.get_next_value().squeeze() 
            else:                                               # Other timestep
                next_non_terminal = 1.0 - dones[:, t + 1]
                next_value = values[:, t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[:, t] + self.gamma * next_value * next_non_terminal - values[:, t]

            # GAE Advantage
            gae = delta + self.gamma * self.lam * next_non_terminal * gae

            advantages[:, t] = gae

            # Target for critic update
            value_target[:,t] = advantages[:,t] + values[:,t]
        

        return advantages, value_target
    
    def actor_update(self, advantages, index) -> None:
        """
        Update actor network
        """

        # Loss evaluation
        loss = self.ppo_loss(advantages)

        if self.writer is not None:
            self.writer.add_scalar("Loss", loss, index)

        # Store old network
        self.buffer.store_network(self.actor)

        # Backpropagation
        self.a_optim.zero_grad()
        loss.backward() 
        self.a_optim.step()

    
    def critic_update(self, value_target) -> None:
        """
        Use value_target to eval MSE and update NN param
        """

        # Get value estimated from the buffer
        value_estimated = self.buffer.get_value()

        # Eval mse loss 
        value_loss = torch.nn.functional.mse_loss(value_estimated, value_target)

        # Backpropagation
        self.c_optim.zero_grad()
        value_loss.backward()
        self.c_optim.step()