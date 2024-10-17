import gymnasium as gym
from time import sleep
import random

n_env = 2
running = True

def make_env():
    return gym.make("CartPole-v1")

envs = gym.vector.SyncVectorEnv([make_env for _ in range(n_env)])

_,_ = envs.reset()

try:
    while running:
        action = [random.randint(0,1), random.randint(0,1)]

        print("Action: ", action)

        nex_obs, reward, term, trunc, _ = envs.step(action)
        print("State ", nex_obs)
        print("Reward ", reward)
        print("Terminated", term)
        print("Trunc", trunc)
        done = term | trunc
        print("Done: ", done)

        for d in done:
            if d:
                print("Vediamo cosa succede")
        
        sleep(0.5)
except KeyboardInterrupt:
    running = False
