#!/bin/bash
# train_mm.sh - Train Markov Models without Makefile
# Usage: ./train_mm.sh <n_states> [seed] [--wandb]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to display help
show_help() {
    echo "Usage: $0 <n_states> [seed] [options]"
    echo ""
    echo "Train a Markov Model with specified number of states and random seed."
    echo ""
    echo "Arguments:"
    echo "  n_states    Number of Markov model states (e.g., 10, 100, 1000)"
    echo "  seed        Random seed for reproducibility (default: 42)"
    echo ""
    echo "Options:"
    echo "  --wandb_log=True    Enable Weights & Biases logging (disabled by default)"
    echo "  --wandb_log=False   Explicitly disable Weights & Biases logging"
    echo "  --sparse N          Use sparse model with N% sparsity (e.g., --sparse 75)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 100                       # Train MM-100 with default seed 42"
    echo "  $0 100 123                   # Train MM-100 with seed 123"
    echo "  $0 100 123 --wandb_log=True  # Train with WandB logging enabled"
    echo "  $0 100 42 --sparse 75        # Train sparse model (75% zero transitions)"
    echo ""
    echo "NOTE: This script assumes data has already been generated. To generate data, use:"
    echo "  python scripts/mm_generate.py <n_states> --train <n_train> --val <n_val> [--sparsity N]"
    echo ""
    echo "Data generation examples:"
    echo "  python scripts/mm_generate.py 10 --train 10000 --val 1000"
    echo "  python scripts/mm_generate.py 100 --train 50000 --val 5000"
    echo "  python scripts/mm_generate.py 100 --train 50000 --val 5000 --sparsity 75"
}

# Parse command line arguments
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 1
fi

N_STATES=$1
SEED=42  # Default seed

# Check if second argument is a number (seed) or an option
if [ $# -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]]; then
    SEED=$2
    shift 2
else
    shift 1
fi

SPARSITY=""
EXTRA_ARGS=""  # Arguments to pass through to train_mm.py

# Parse optional arguments
while [ $# -gt 0 ]; do
    case $1 in
        --wandb_log=*)
            # Pass through wandb_log argument
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        --sparse)
            SPARSITY=$2
            shift 2
            ;;
        --*)
            # Pass through any other arguments starting with --
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if ! [[ "$N_STATES" =~ ^[0-9]+$ ]]; then
    print_error "n_states must be a positive integer"
    exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    print_error "seed must be a positive integer"
    exit 1
fi

if [ -n "$SPARSITY" ] && ! [[ "$SPARSITY" =~ ^[0-9]+$ ]]; then
    print_error "sparsity must be a positive integer (percentage)"
    exit 1
fi

# Set model name based on sparsity
if [ -n "$SPARSITY" ]; then
    MODEL_NAME="MM-${N_STATES}-sparse-${SPARSITY}"
    DATA_DIR="data/mm${N_STATES}-sparse-${SPARSITY}"
else
    MODEL_NAME="MM-${N_STATES}"
    DATA_DIR="data/mm${N_STATES}"
fi

print_info "Training configuration:"
print_info "  Model: $MODEL_NAME"
print_info "  Seed: $SEED"
print_info "  Data directory: $DATA_DIR"
if [ -n "$EXTRA_ARGS" ]; then
    print_info "  Extra arguments:$EXTRA_ARGS"
fi

# Check if data exists
if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
    print_error "Data not found in $DATA_DIR"
    print_error "Please generate data first using:"
    if [ -n "$SPARSITY" ]; then
        if [ "$N_STATES" -le 10 ]; then
            print_error "  python scripts/mm_generate.py $N_STATES --train 10000 --val 1000 --sparsity $SPARSITY"
        else
            print_error "  python scripts/mm_generate.py $N_STATES --train 50000 --val 5000 --sparsity $SPARSITY"
        fi
    else
        if [ "$N_STATES" -le 10 ]; then
            print_error "  python scripts/mm_generate.py $N_STATES --train 10000 --val 1000"
        else
            print_error "  python scripts/mm_generate.py $N_STATES --train 50000 --val 5000"
        fi
    fi
    exit 1
fi

print_success "Data found in $DATA_DIR"

# Step 2: Create directory structure
TRAIN_DIR="trainings/MM/$MODEL_NAME/$MODEL_NAME-$SEED"
print_info "Creating directory structure: $TRAIN_DIR"
mkdir -p "$TRAIN_DIR"

# Step 3: Generate configuration
print_info "Generating configuration..."
CONFIG_FILE="$TRAIN_DIR/config.py"

# Generate config with appropriate model ID
if [ -n "$SPARSITY" ]; then
    python configs/mm_config.py "${N_STATES}-sparse-${SPARSITY}" "$SEED" > "$CONFIG_FILE"
else
    python configs/mm_config.py "$N_STATES" "$SEED" > "$CONFIG_FILE"
fi

# No need to modify config - just pass arguments to train_mm.py

# Step 4: Create training start marker
print_info "Creating training markers..."
START_FILE="$TRAIN_DIR/training.start"
echo "Start: $(date)" > "$START_FILE"
echo "Config: mm${N_STATES}${SPARSITY:+-sparse-}${SPARSITY}-s${SEED}" >> "$START_FILE"
echo "Status: TRAINING" >> "$START_FILE"

# Step 5: Run training
LOG_FILE="$TRAIN_DIR/training.log"
print_info "Starting training (logging to $LOG_FILE)..."
print_info "You can monitor progress with: tail -f $LOG_FILE"

# Run training and capture exit code
set +e  # Temporarily disable exit on error
python -u train_mm.py "$CONFIG_FILE" $EXTRA_ARGS > "$LOG_FILE" 2>&1
TRAINING_EXIT_CODE=$?
set -e  # Re-enable exit on error

# Step 6: Create completion marker
DONE_FILE="$TRAIN_DIR/model.done"
echo "Completed: $(date)" > "$DONE_FILE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS" >> "$DONE_FILE"
    
    # Extract final loss from log
    FINAL_LOSS=$(grep -o "step [0-9]*.*loss [0-9.]*" "$LOG_FILE" | tail -1)
    if [ -n "$FINAL_LOSS" ]; then
        echo "$FINAL_LOSS" >> "$DONE_FILE"
    else
        echo "Final loss: unknown" >> "$DONE_FILE"
    fi
    
    print_success "Training completed successfully!"
    print_success "Results saved in: $TRAIN_DIR"
    
    # Show final metrics if available
    if [ -n "$FINAL_LOSS" ]; then
        print_info "Final training metrics: $FINAL_LOSS"
    fi
    
    # Check if checkpoint was saved
    if [ -f "$TRAIN_DIR/ckpt.pt" ]; then
        print_success "Model checkpoint saved: $TRAIN_DIR/ckpt.pt"
    fi
else
    echo "Status: FAILED (exit code: $TRAINING_EXIT_CODE)" >> "$DONE_FILE"
    print_error "Training failed with exit code: $TRAINING_EXIT_CODE"
    print_error "Check the log file for details: $LOG_FILE"
    
    # Show last few lines of log for quick debugging
    echo ""
    print_error "Last 10 lines of training log:"
    tail -10 "$LOG_FILE"
    exit 1
fi

# Optional: Show how to evaluate the model
echo ""
print_info "To evaluate this model, run:"
print_info "  python scripts/mm_eval.py --ckpt_path $TRAIN_DIR/ckpt.pt"