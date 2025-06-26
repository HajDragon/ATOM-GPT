# Train a GPT model on DarkLyrics metal music dataset
# Optimized for RTX 3050 (8GB VRAM)

out_dir = 'out-darklyrics'
eval_interval = 500  # Evaluate every 500 iterations
eval_iters = 100     # Use fewer evaluation iterations to save time
log_interval = 10    # Log every 10 iterations

# Checkpoint settings
always_save_checkpoint = True  # Save checkpoints for resume capability
init_from = 'scratch'  # Start from scratch

# Weights & Biases logging (optional)
wandb_log = False
wandb_project = 'darklyrics-gpt'
wandb_run_name = 'metal-lyrics'

# Dataset
dataset = 'DarkLyrics'

# RTX 3050 Optimized Settings (8GB VRAM)
gradient_accumulation_steps = 4  # Simulate larger batch size
batch_size = 8                   # Small batch size for RTX 3050
block_size = 512                 # Context length - reduced for memory

# Model architecture - Small but capable model for RTX 3050
n_layer = 8          # 8 transformer layers
n_head = 8           # 8 attention heads  
n_embd = 512         # 512 embedding dimensions
dropout = 0.1        # Low dropout for better learning

# Training settings
learning_rate = 3e-4      # Standard learning rate
max_iters = 10000         # Total training iterations
lr_decay_iters = 10000    # Decay learning rate over training
min_lr = 3e-5            # Minimum learning rate (learning_rate / 10)

# Optimization
beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1
grad_clip = 1.0       # Gradient clipping

# System settings
device = 'cuda'       # Use GPU
dtype = 'float16'     # Use float16 for RTX 3050 efficiency
compile = False       # Disable compilation for compatibility
