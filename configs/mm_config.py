#!/usr/bin/env python3
"""Generate minimal training config for MM experiments"""
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python mm_config.py N_STATES [SEED] or MM_ID [SEED]")
    sys.exit(1)

# Parse first argument - could be N_STATES or MM_ID (like "100-sparse-25")
mm_id = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# Parse MM_ID to extract n_states and variants
if '-sparse-' in mm_id:
    # Handle sparse variants like "100-sparse-25"
    parts = mm_id.split('-')
    n_states = int(parts[0])
    sparsity = int(parts[2])
    is_sparse = True
    data_suffix = f"-sparse-{sparsity}"
else:
    # Handle regular variants like "100"
    n_states = int(mm_id)
    is_sparse = False
    data_suffix = ""

# Model size based on n_states
if n_states <= 10:
    n_layer, n_embd = 2, 128
elif n_states <= 100:
    n_layer, n_embd = 2, 256  
elif n_states <= 1000:
    n_layer, n_embd = 4, 512
else:
    n_layer, n_embd = 4, 768

# Training iterations based on complexity - using 5K for good balance
if n_states <= 10:
    max_iters = 1000  # MM-10 convergence
elif n_states <= 100:
    max_iters = 5000  # Good balance for MM-100
else:
    max_iters = 5000  # MM-1000 needs more training

config = f"""# Minimal MM-{mm_id} Config (seed {seed})
out_dir = 'trainings/MM/MM-{mm_id}/MM-{mm_id}-{seed}'
eval_interval = 500
log_interval = 10
eval_iters = 50
always_save_checkpoint = True

# Data
dataset = 'openwebtext'  # Dummy dataset type
data_dir = f'data/mm{n_states}{data_suffix}'

# Model
n_layer = {n_layer}
n_head = 8
n_embd = {n_embd}
dropout = 0.0
bias = False
vocab_size = {n_states + 1}  # Include BOS token
block_size = {256 if n_states > 10 else 128}

# Training  
batch_size = {8 if n_states > 10 else 16}
max_iters = {max_iters}
learning_rate = 5e-3
weight_decay = 1e-1
decay_lr = True
warmup_iters = 200
lr_decay_iters = {max_iters}
min_lr = 1e-4

# System
device = 'auto'
dtype = 'auto'
compile = False
wandb_log = True
wandb_project = 'MM-experiments'
wandb_run_name = 'MM-{mm_id}-{seed}'

# MM-specific: No completion generation needed

# Seed for reproducibility  
train_seed = {seed}
"""

print(config)
