import torch
import gymnasium as gym
import Network
from parameters import *

# Enviroment
#env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()

# Network_id
folder = "NN/"
name = "pole_002"
ext = "_Actor.pth"

path = folder + name + ext

# Network
actor = Network.Actor(len(state), 2, path)

running = True
ep_rew = 0

try: 
    while running:
        
        q = actor(state)
        action = torch.argmax(q)

        next_state, reward, term, trunc, _ = env.step(int(action))

        ep_rew += reward

        if term or trunc:
            state, _ = env.reset()

            print(f"\rEpisode reward: {ep_rew}", end="")
            ep_rew = 0

        state = next_state


except KeyboardInterrupt:
    running = False