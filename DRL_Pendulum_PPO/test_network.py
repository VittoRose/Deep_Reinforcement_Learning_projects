import Enviroment
import Graphics
import Network
import torch
from time import time

# Test enviroment
test_env = Enviroment.PendolumEnv()

# Load network parameters
path = "NN/Test0_Actor.pth"
agent = Network.Actor(test_env.state_size, test_env.action_size, path)

# Grapichs management
gui = Graphics.GUI()
gui.GUI_init(time())

running = True

state = test_env.reset()
ep_reward, reset = 0, 0

try: 
    while running:

        running = gui.GUI_quit()

        # Select the action from the network
        with torch.no_grad():
            q = agent(state)

        action = torch.argmax(q)
        
        # Perform a step in the enviroment
        state, reward, terminated, truncated = test_env.step(action)
        ep_reward += reward

        # Reset the enviroment
        if truncated or terminated:
            state = test_env.reset()

            ep_reward = 0
            reset =+ 1

        ang = test_env.ang

        gui.draw(ang, reset, ep_reward)

except KeyboardInterrupt:
    running = False
