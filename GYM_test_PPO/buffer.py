import torch

class Buffer():
    def __init__(self, n_step, n_env, envs):
                
        self.obs = torch.zeros((n_step, n_env) + envs.single_observation_space.shape)
        self.actions = torch.zeros((n_step, n_env) + envs.single_action_space.shape)
        self.logprobs = torch.zeros((n_step, n_env))
        self.rewards = torch.zeros((n_step, n_env))
        self.dones = torch.zeros((n_step, n_env))
        self.values = torch.zeros((n_step, n_env))

        