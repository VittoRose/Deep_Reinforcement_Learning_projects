import DNQ
import Enviroment
from time import time
import random
import Jumping_bird as jb
import torch

random.seed(time())
env = Enviroment.FlappyBird()
screen = jb.Graphics()

# Load old network
agent = DNQ.Agent(env.state_size, env.action_size,"NN/Buffer100k_Batch128_pc_lab.pth")

state = env.reset()
reset = 0
running = True

try:
    while running:

        running, _ = screen.user_interaction()

        with torch.no_grad():
            q = agent(state)

        action = torch.argmax(q)

        state, reward, terminated, truncated = env.step(action)

        reward += reward

        if terminated or truncated:
            reset +=1
            reward = 0

        bird, pipes = env.get_disp_data()

        screen.draw(bird, pipes, reset, env.score, reward)
except KeyboardInterrupt:
    running = False