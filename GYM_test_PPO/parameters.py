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
T = 64
K = 4
MINI_BATCH = 64//K

# Loop parameters
MAX_ITERATION = 10_000
TEST_RESET = 50