from parameters import *
import numpy as np
import torch

class DictBuffer():
    def __init__(self, state_size, n_envs, capacity=BUFFER_SIZE, batch_size=BATCH_SIZE) -> None:
        """
        Param:
        - observation size -> enviroment state dimension
        - n_envs -> number of evniroment currently working
        """
        # Param from enviroment
        self.state_size = state_size
        self.n_env = n_envs

        # List to store cumulative reward and episode length
        self.rew = [0 for _ in range(self.n_env)]
        self.ep_len = [0 for _ in range(self.n_env)]

        # Buffer dimension
        self.capacity = capacity
        self.batch_size = batch_size

        self.reset()

    def reset(self):
        """
        Preallocation for buffer
        """
        
        # Create a list to store lenght and reward of each enviroment 
        #self.episode_rews = [0 for _ in range(self.n_envs)]
        #self.episode_lens = [0 for _ in range(self.n_envs)]

        # Preallocation for buffer
        self.buffer = dict(state=np.zeros((self.capacity, self.state_size)),          # (capacity)_rows x (state_size)_columns
                        reward = np.zeros(self.capacity),
                        action=np.zeros(self.capacity),
                        next_state=np.zeros((self.capacity, self.state_size)),
                        terminated=np.zeros(self.capacity),
                        truncated=np.zeros(self.capacity))
        
        # Init index and size to count buffer length
        self.i = 0
        self.size = 0

    def store_data(self, state, reward, action, next_state, terminated, truncated) -> tuple[list[int], list[int]]:
        """
        Store data in the buffer
        """

        # List that contain cumulative rewards and episode lenght for each enviroment
        rewards = []
        ep_length = []

        for j in range(self.n_env):

            # Circular index
            index = (self.i + j) % self.capacity         
    
            # Copy data input in the first avaible row
            self.buffer["state"][index] = state[j]
            self.buffer["reward"][index] = reward[j]
            self.buffer["action"][index] = action[j]
            self.buffer["next_state"][index] = next_state[j]
            self.buffer["terminated"][index] = terminated[j]
            self.buffer["truncated"][index] = truncated[j]

            # Store cumulative reward
            self.rew[j] += reward[j]
            self.ep_len[j] += 1

            # If episode terminate store the value
            if terminated[j] or truncated[j]:

                # Output variable lists of reward and episode length
                rewards.append(self.rew[j])
                ep_length.append(self.ep_len[j])

                # Clear enviroment reward and length
                self.rew[j], self.ep_len[j] = 0,0
            
        # Update index, if capacity exeeded clear old transition
        self.i = (self.i + self.n_env) % self.capacity
        
        self.size = min(self.size+1, self.capacity)

        return rewards, ep_length

    def sample(self) -> dict:
        """
        Return a batch from buffer data
        """
        
        # Create an array with the index to sample from the buffer
        index = np.random.choice(self.size, self.batch_size)

        batch = dict(
            state=np.zeros((self.batch_size, self.state_size)),
            reward=np.zeros(self.batch_size),
            action=np.zeros(self.batch_size),
            next_state=np.zeros((self.batch_size, self.state_size)),
            terminated=np.zeros(self.batch_size),
            truncated= np.zeros(self.batch_size))
        
        # Copy experience transition from buffer to batch
        for batch_i, buffer_i in enumerate(index):

            batch["state"][batch_i] = self.buffer["state"][buffer_i]
            batch["reward"][batch_i] = self.buffer["reward"][buffer_i]
            batch["action"][batch_i] = self.buffer["action"][buffer_i]
            batch["next_state"][batch_i] = self.buffer["next_state"][buffer_i]
            batch["terminated"][batch_i] = self.buffer["terminated"][buffer_i]
            batch["truncated"][batch_i] = self.buffer["truncated"][buffer_i]

        return batch
    
    def show(self):
        """
        Print the buffer on a txt
        """
        with open("buffer_out.txt", "w") as file:
            file.write(str(self.buffer))


    def __len__(self):
        """
        Lenght = last index with data
        """
        return self.size
    
class Collector: 
    def __init__(self, policy, env, buffer, action_size, logger=None, exploration : bool = True ):
        """
        Collector initiazlizer: allow the interaction between agent and multiple enviroment
        The data collected from the enviroments are stored in the ReplayBuffer
        """

        # Vector enviroment object
        self.env = env

        # Hyperparameters
        self.exploration = exploration
        self.policy = policy
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

    def collect(self, n_res : int = None , n_exp : int = None  ):
        """
        Collect transitions from the interaction policy-enviroment and store in the buffer
        :param n_ep: number of episodes to collect (an episode is counted if terminated or truncated is true)
        """

        # Inizialise counter
        count_res = 0
        count_exp = 0

        while True:

            with torch.no_grad():
                # Get action values from a single or multiple observations
                q_val = self.policy.action(self.state)

            # Select the max activation for each q_value
            actions = np.asarray(torch.argmax(q_val, dim=1))

            # Select random action for exploration if needed
            if self.exploration:
                random_sample = np.random.rand()
                if random_sample < self.policy.eps:
                    actions = np.random.choice(range(self.action_size), len(actions))

            # Perform a step in all enviroment
            next_state, reward, terminated, truncated = self.env.step(actions)

            # Add data in replay buffer
            rewards, ep_length = self.buffer.store_data(self.state, reward, actions, next_state, terminated, truncated)

            if np.any(truncated | terminated):
                to_reset = np.where(truncated | terminated)[0]
                
                if self.writer is not None:
                    for index, i in enumerate(rewards):
                        self.writer.add_scalar("Train/Reward per episode", i, self.reset_num + index)
                    for index,j in enumerate(ep_length):
                        self.writer.add_scalar("Train/Episode length", j, self.reset_num + index)

                # Count the number of enviroment reset, total and partial
                self.reset_num += len(to_reset)
                count_res += len(to_reset)

                # Reset the enviroment terminated or truncated
                self.state[to_reset] = self.env.reset(to_reset)


            # Update the state for the next loop
            self.state = next_state

            count_exp += len(self.env)

            # Exit loop if collected enough sample
            if count_exp > n_exp:
                break
            
        return self.reset_num, 