#!/usr/bin/env python3
"""Generate configs for MM-extra experiments"""
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python mm_config_extra.py VARIANT [SEED] [DATA_SUFFIX]")
    print("Variants: c1 (1L2H256D), c2 (1L1H128D), c3 (1L1H101D), c4 (1L1H101D+one-hot), c5 (1L MLP-only+one-hot), c6 (1L MLP-only+one-hot+1x-ratio)")
    print("New: c7 (c4+pos_ape=False), c8 (c7+RELU), c9 (c8+identity_first), c10 (c9+use_ln=False)")
    print("Extra: c11 (c9+mlp_only), c12 (c10+mlp_only)")
    print("DATA_SUFFIX: Optional suffix for data dir (e.g., '-sparse-75' -> mm100-sparse-75)")
    sys.exit(1)

variant = sys.argv[1].lower()
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
data_suffix = sys.argv[3] if len(sys.argv) > 3 else ""

# Architecture variants for MM-100 extra experiments
variants = {
    'c1': {
        'n_layer': 1, 'n_head': 2, 'n_embd': 256,
        'max_iters': 3000, 'learning_rate': 5e-3,
        'description': 'Reduced layers + heads'
    },
    'c2': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 128,
        'max_iters': 3000, 'learning_rate': 5e-3,
        'description': 'Minimal viable transformer RELU',
        'mlp_filter': 'RELU'
    },
    'c3': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'One-hot frozen  RELU',
        'mlp_filter': 'RELU',
        'one_hot': True
    },
    'c4': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'One-hot RELU + 1x MLP ratio',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1
    },
    'c5': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'MLP-only + one-hot RELU',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_only': True
    },
    'c6': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'MLP-only + one-hot + 1x MLP ratio',
        'one_hot': True,
        'mlp_ratio': 1,
        'mlp_only': True
    },
    'c7': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'c4 + no positional embeddings',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1,
        'pos_ape': False
    },
    'c8': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'c4 + no LayerNorm',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1,
        'use_ln': False
    },
    'c9': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'c4 + no LayerNorm + no pos embeddings',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1,
        'use_ln': False,
        'pos_ape': False
    },
    'c10': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'c9 + identity first matrix',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1,
        'pos_ape': False,
        'mlp_first_matrix_identity': True
    },
    'c11': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'c9 + no attention (MLP-only)',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1,
        'use_ln': False,
        'pos_ape': False,
        'mlp_only': True
    },
    'c12': {
        'n_layer': 1, 'n_head': 1, 'n_embd': 101,
        'max_iters': 5000, 'learning_rate': 5e-3,
        'description': 'c10 + no attention (MLP-only)',
        'mlp_filter': 'RELU',
        'one_hot': True,
        'mlp_ratio': 1,
        'use_ln': False,
        'pos_ape': False,
        'mlp_first_matrix_identity': True,
        'mlp_only': True
    }
}

if variant not in variants:
    print(f"Error: Unknown variant '{variant}'. Use c1-c12")
    sys.exit(1)

params = variants[variant]

# Determine output directory and data directory based on suffix
if data_suffix:
    # Sparse variant: trainings/MM/MM-100-sparse-75-extra/MM-100-sparse-75-extra-c12-s42
    base_name = f"MM-100{data_suffix}-extra"
    out_dir = f"trainings/MM/{base_name}/{base_name}-{variant}-s{seed}"
    data_dir = f"data/mm100{data_suffix}"
    config_name = f"MM-100{data_suffix}-extra {variant.upper()}"
else:
    # Regular variant: trainings/MM/MM-100-extra/MM-100-extra-c12-s42
    out_dir = f"trainings/MM/MM-100-extra/MM-100-extra-{variant}-s{seed}"
    data_dir = "data/mm100"
    config_name = f"MM-100-extra {variant.upper()}"

config = f"""# {config_name} Config (seed {seed})
# {params['description']}
out_dir = '{out_dir}'
eval_interval = 500
log_interval = 10
eval_iters = 50
always_save_checkpoint = True

# Data
dataset = 'openwebtext'  # Dummy dataset type
data_dir = '{data_dir}'

# Model Architecture - {params['description']}
n_layer = {params['n_layer']}
n_head = {params['n_head']}
n_embd = {params['n_embd']}
dropout = 0.0
bias = False
vocab_size = 101  # MM-100 + BOS token
block_size = 256
one_hot_embed = {params.get('one_hot', False)}
mlp_only = {params.get('mlp_only', False)}
mlp_ratio = {params.get('mlp_ratio', 4)}
pos_ape = {params.get('pos_ape', True)}
mlp_filter = '{params.get('mlp_filter', 'GELU')}'
mlp_first_matrix_identity = {params.get('mlp_first_matrix_identity', False)}
use_ln = {params.get('use_ln', True)}

# Training (adjusted for minimal architectures)
batch_size = 8  # Smaller batch for parallel execution
max_iters = {params['max_iters']}
learning_rate = {params['learning_rate']}
weight_decay = 1e-1
decay_lr = True
warmup_iters = 400
lr_decay_iters = {params['max_iters']}
min_lr = 1e-4

# System
device = 'auto'
dtype = 'auto'
compile = False
wandb_log = False
wandb_project = 'MM-experiments'
wandb_run_name = 'MM-extra-{variant}-s{seed}'

# MM-specific: No completion generation needed

# Seed for reproducibility
train_seed = {seed}
"""

print(config)
