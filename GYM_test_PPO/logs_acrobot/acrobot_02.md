# Enviroment: Acrobot-v1
Seed: 92, (deterministic)
## Training parameters
- Total epoch: 2000
- Number of enviroments: 4
- Timestep for collecting data T = 128
- Total data for each loop: 512
- Update epoch K = 4
- Minibatch size 128

## Hyperparameters
* Discount factor: 0.99
* GAE lambda: 0.9
* Learning rate: 0.00025
* Clipping factor: 0.2
* Loss: c1 = 0.5; c2 = 0.01
