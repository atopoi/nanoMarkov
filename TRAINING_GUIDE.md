# Training Guide

This guide explains how to train Markov Models.

## Two-Step Process

### Step 1: Generate Data

First, generate the Markov Model data using the appropriate parameters:

```bash
# For MM-10 (small model)
python scripts/mm_generate.py 10 --train 10000 --val 1000

# For MM-100 (medium model)
python scripts/mm_generate.py 100 --train 50000 --val 5000

# For MM-1000 (large model) (might take a while to generate)
python scripts/mm_generate.py 1000 --train 50000 --val 5000

# For sparse models (e.g., 75% zero transitions)
python scripts/mm_generate.py 100 --train 50000 --val 5000 --sparsity 75
```

Data will be saved in:
- `data/mm{n_states}/` for regular models
- `data/mm{n_states}-sparse-{sparsity}/` for sparse models

### Step 2: Train the Model

Use the provided `train_mm.sh` script:

```bash
# Basic usage (default seed 42, wandb disabled by default)
./train_mm.sh 100

# Specify custom seed
./train_mm.sh 100 123

# Enable WandB logging
./train_mm.sh 100 123 --wandb_log=True

# Train sparse model (assumes data already generated)
./train_mm.sh 100 42 --sparse 75
```

#### Directory Structure

The training process creates the following structure:
```
trainings/MM/
└── MM-{n_states}/
    └── MM-{n_states}-{seed}/
        ├── config.py          # Training configuration
        ├── training.start     # Training start marker
        ├── training.log       # Training output log
        ├── ckpt.pt           # Model checkpoint
        └── model.done        # Completion marker with final loss
```

### Manual Process

If you prefer to run commands directly (you still need to create the training data first):

```bash
# Example: Train MM-100 with seed 123

# 1. Create directory structure
mkdir -p trainings/MM/MM-100/MM-100-123/

# 2. Generate configuration
python configs/mm_config.py 100 123 > trainings/MM/MM-100/MM-100-123/config.py

# 3. (Optional) Enable WandB by editing the config file
# Change "wandb_log = False" to "wandb_log = True" in the config

# 4. Run training
python -u train_mm.py trainings/MM/MM-100/MM-100-123/config.py > trainings/MM/MM-100/MM-100-123/training.log 2>&1
```

### Alternative: Override WandB via Command Line

Instead of editing the config file, you can override wandb setting:

```bash
# Enable wandb via command line
python train_mm.py trainings/MM/MM-100/MM-100-123/config.py --wandb_log=True
```


## Examples

### Example 1: Train MM-10 with default settings
```bash
# Generate data
python scripts/mm_generate.py 10 --train 10000 --val 1000

# Train
./train_mm.sh 10
```

### Example 2: Train MM-100 with custom seed and WandB
```bash
# Generate data (if not already done)
python scripts/mm_generate.py 100 --train 50000 --val 5000

# Train with seed 456 and WandB
./train_mm.sh 100 456 --wandb_log=True
```

### Example 3: Train sparse MM-100 (75% sparsity)
```bash
# Generate sparse data
python scripts/mm_generate.py 100 --train 50000 --val 5000 --sparsity 75

# Train sparse model
./train_mm.sh 100 42 --sparse 75
```

## Monitoring Training

To monitor training progress:
```bash
# Watch the log file
tail -f trainings/MM/MM-100/MM-100-42/training.log
```

## After Training

Once training completes:

1. The model checkpoint will be saved as `ckpt.pt`
2. Final loss will be recorded in `model.done`
3. To evaluate the model:
   ```bash
   python scripts/mm_eval.py --ckpt_path trainings/MM/MM-100/MM-100-42/ckpt.pt
   ```
4. It's also possible to resume a training from a checkpoint - see train_mm.py
