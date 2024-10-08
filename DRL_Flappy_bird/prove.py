import torch
import numpy as np

import DNQ

NN = DNQ.Agent()

state = torch.rand((40, NN.state_size))

print(state)

with torch.no_grad():
    q_val = NN(state)

print("Q_val:")

print(q_val)

print("Actions")

actions = torch.argmax(q_val, dim=1)

print(actions)

print("As array")
array = np.asarray(torch.argmax(q_val, dim=1))

print(array)