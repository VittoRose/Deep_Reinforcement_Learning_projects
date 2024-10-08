from parameters import *
import torch
from copy import deepcopy

class Buffer():
    def __init__(self, actor, n_envs, capacity=BUFFER_SIZE) -> None:
        """
        Param:
        - observation size -> enviroment state dimension
        - n_envs -> number of evniroment currently working
        """
        # Param from enviroment
        self.n_env = n_envs

        # List to store cumulative reward and episode length
        self.rew = [0 for _ in range(self.n_env)]
        self.ep_len = [0 for _ in range(self.n_env)]

        # Buffer dimension
        self.capacity = capacity

        # Old network store
        self.old_NN = deepcopy(actor)

        self.reset()

    def reset(self):
        """
        Create empty data structures to store data
        All data structures are 
        """

        # State buffer
        self.state = []

        # Reward buffer
        self.rewards = torch.zeros((self.n_env, self.capacity))

        # Buffer V(s_t) 
        self.values = torch.zeros((self.n_env, self.capacity))

        # Flag Buffer 
        self.dones = torch.zeros((self.n_env, self.capacity))
        self.trunc = torch.zeros((self.n_env, self.capacity))

        # Next value buffer
        self.next_value = torch.zeros(self.n_env)

        # Action / probability action buffer
        self.action = torch.zeros((self.n_env, self.capacity))
        self.prob = torch.zeros((self.n_env, self.capacity))
        self.old_prob = torch.zeros((self.n_env, self.capacity))

    def store_transition(self, state, reward, terminated, truncated, value, timestamp) -> tuple[list[int], list[int]]:
        """
        Store data in the buffer
        """

        # List that contain cumulative rewards and episode lenght for each enviroment
        #rewards = []
        #ep_length = []

        # Store data 
        self.state.append(state)
        self.rewards[:,timestamp] = torch.as_tensor(reward)
        self.dones[:,timestamp] = torch.as_tensor(terminated)
        self.trunc[:,timestamp] = torch.as_tensor(truncated)
        self.values[:,timestamp] = value.squeeze()

        """
        # If episode terminate store the value
        if terminated[j]:

            # Output variable lists of reward and episode length
            rewards.append(self.rew[j])
            ep_length.append(self.ep_len[j])

            # Clear enviroment reward and length
            self.rew[j], self.ep_len[j] = 0,0
        """        

        #return rewards, ep_length

    def get_transition(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Return all the variable stored during the interaction policy-enviroment
        """
        
        return self.rewards, self.dones, self.values
    
    def store_next_value(self, next_value) -> None:
        """
        Store the next value V(s_{t+1}) at the end of the collectin cycle
        """
        self.next_value = next_value

    def get_next_value(self) -> torch.tensor:
        """
        Return V(s_{t+1})
        """
        return self.next_value
    
    def get_value(self) -> torch.tensor:
        """
        Return V(s_t)
        """
        return self.values

    def store_action(self, action, prob, timestamp) -> None:
        self.action[:,timestamp] = action
        self.prob[:,timestamp] = prob.squeeze()

    def get_prob(self) -> list[torch.tensor]:
        """
        Return softmax(pi_{theta_old}(s,a))
        """
        return self.prob
    
    def store_network(self, network) -> None:
        """
        Copy network parameters from input network to buffer
        """

        self.old_NN.load_state_dict(network.state_dict())

    def old_network(self, state, actions, timestamp) -> None:
        """
        Get a state as an input, evaluate and store the probability function with the old network
        """
        q_val = self.old_NN(state)

        action_probs = torch.softmax(q_val, dim=-1)

        # Store the old policy probability for the action chosen
        self.old_prob[:,timestamp] = torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze()