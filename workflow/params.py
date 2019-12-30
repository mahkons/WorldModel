

# image
image_height = 96
image_width = 96

# MDNRNN
# z_size = 32 # CarRacing
z_size = 8 # LunarLander
n_hidden = 256
n_gaussians = 5

# Agent
actor_lr = 1e-4
critic_lr = 1e-4
mem_size = 20 * 1000

GAMMA = 0.99
BATCH_SIZE = 64
TAU = 0.002

# memory
memtype = "Prioritized"
PRIORITY_EPS = 0.005
PRIORITY_ALPHA = 0.5
