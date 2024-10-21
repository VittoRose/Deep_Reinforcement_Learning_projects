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
* Discount factor: 0.9
* GAE lambda: 1
* Learning rate: 0.00025
* Clipping factor: 0.15
* Loss: c1 = 0.5; c2 = 0.0
