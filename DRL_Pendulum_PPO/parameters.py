from numpy import pi

# Actor param
LR_ACTOR = 1e-3

# Critic param
LR_CRITIC = 1e-3

# Clipping factor
CLIP = 0.2

# Discount factor
GAMMA = 0.98
LAMBDA = 0.95

# Buffer param
BUFFER_SIZE = 64       # it's also the number of timestep explored for each iteration

# Boundaries
MAX_ANGLE = pi/2
MAX_OMEGA = 2
SUCCESS_STEP = 500
SMALL_ANGLE = 0.05

# Reward
MAX_ANGLE_EXCEEDED = -20
SMALL_ANGLE_REWARD = 1
