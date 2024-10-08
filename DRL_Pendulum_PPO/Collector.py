from parameters import *
import torch
import numpy as np

class Collector:

    def __init__(self, trainer, env, buffer, action_size, logger=None):
        """
        Collector initiazlizer: allow the interaction between agent and multiple enviroment
        The data collected from the enviroments are stored in the ReplayBuffer
        """

        # Vector enviroment object
        self.env = env

        # Hyperparameters
        self.trainer = trainer
        self.buffer = buffer
        self.action_size = action_size

        # Init for initial state
        self.state = None

        # Tensorboard writer
        self.writer = logger

        # Nubmer of reset
        self.reset_num = 0

        # Initialize the enviroment
        self.reset()

    def reset(self):
        """
        Reset the enviroment and store the first state in the auxiliary buffer
        """
        self.state = self.env.reset()

    def collect(self, n_exp : int = None):
        """
        Collect transitions from the interaction policy-enviroment and store in the buffer
        :param n_exp: nuber of timesteps to collect experience in each enviroment
        """

        for count_exp in range(0,n_exp):
            
            # Get activation from previous state
            q_val = self.trainer.actor(self.state)

            # Map q_val in probability value
            action_probs = torch.softmax(q_val, dim=-1)

            # Create a probability distribution from the NN output
            distribution = torch.distributions.Categorical(action_probs)

            # Sample the actions from the distribution
            actions = distribution.sample()
            prob_chosen = torch.gather(action_probs, 1, actions.unsqueeze(1))

            # Store action and probability associated
            self.buffer.store_action(actions, prob_chosen, count_exp)            

            # Evaluate the state with the old network
            self.buffer.old_network(self.state, actions, count_exp) 
            
            # Perform a step in all enviroment
            next_state, reward, terminated, truncated = self.env.step(actions)

            # Value approximation
            value = self.trainer.critic(self.state)

            # Add data in replay buffer
            self.buffer.store_transition(self.state, reward, terminated, truncated, value, count_exp)

            # Manage truncated and terminated enviroment
            if np.any(truncated | terminated):
                to_reset = np.where(truncated | terminated)[0]
                """
                # Add to log each enviroment stats
                if self.writer is not None:
                    for index, i in enumerate(rewards):
                        self.writer.add_scalar("Train/Reward per episode", i, self.reset_num + index)
                    for index,j in enumerate(ep_length):
                        self.writer.add_scalar("Train/Episode length", j, self.reset_num + index)
                """

                # Count the number of enviroment reset
                self.reset_num += len(to_reset)

                # Reset the enviroment terminated or truncated
                self.state[to_reset] = self.env.reset(to_reset)

            # For the last iteration evaluate V(s_{t+1}) and store in the buffer
            if count_exp == n_exp-1:
                next_values = self.trainer.critic(next_state)
                self.buffer.store_next_value(next_values)

            # Update the state for the next loop
            self.state = next_state