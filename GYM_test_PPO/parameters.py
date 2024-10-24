
# Training parameters
n_env = 4
n_step = 128                    # Number of step in the enviroment between each update
BATCH_SIZE = n_env*n_step       # Data collected for each update
MAX_EPOCH = 2_000

# Hyperparameters
LR = 2.5e-4                     # Optimizer learning rate
GAMMA = 0.99                     # Discount factor
GAE_LAMBDA = 0.98               # TD(lambda) factor: 1 -> Monte Carlo; 0 -> TD error
K_EPOCHS = 4                    # Number of update at the end data collection

CLIP = 0.2                      # Clipping factor in policy loss
ENTROPY_COEF = 0.01             # Entropy coefficent for loss calculation
VALUE_COEFF = 0.5               # Value coefficent for loss calculation
VALUE_CLIP = False             # If true -> clip policy loss value

if BATCH_SIZE % K_EPOCHS != 0:
    raise ValueError("Batch size and K_epochs are not compatible")

MINI_BATCH_SIZE = BATCH_SIZE//K_EPOCHS      

# Test parameters
TEST_INTERVAL = 60
RECORD_VIDEO = 100

# Seed
SEED = 92

