# Quick test training on DarkLyrics dataset
# Very small model for testing - trains in minutes

out_dir = 'out-darklyrics-test'
eval_interval = 100  # Evaluate frequently 
eval_iters = 50      # Quick evaluation
log_interval = 5     # Log frequently

# Checkpoint settings
always_save_checkpoint = False  # Don't save for test runs
init_from = 'scratch'

# Weights & Biases logging
wandb_log = False
wandb_project = 'darklyrics-test'
wandb_run_name = 'test-run'

# Dataset
dataset = 'DarkLyrics'

# Very small model for quick testing
gradient_accumulation_steps = 1
batch_size = 4                   # Very small batch
block_size = 256                 # Short context

# Tiny model architecture
n_layer = 4          # Only 4 layers
n_head = 4           # 4 attention heads
n_embd = 256         # 256 embedding dimensions
dropout = 0.1

# Quick training settings
learning_rate = 1e-3      # Higher learning rate for quick training
max_iters = 1000          # Only 1000 iterations for testing
lr_decay_iters = 1000
min_lr = 1e-4

# Optimization
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1
grad_clip = 1.0

# System settings
device = 'cuda'
dtype = 'float16'
compile = False       # Disable compilation for compatibility
