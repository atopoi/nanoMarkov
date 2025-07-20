	#!/usr/bin/env python3
"""Generate MM data - minimal and direct"""
import numpy as np
from hmmlearn import hmm
import argparse
import pickle
import json
import os

def create_sparse_transitions(n_states, sparsity_percent, seed):
    """Create sparse transition matrix with structured sparsity pattern"""
    np.random.seed(seed)
    
    # Calculate number of connections to keep per state
    max_connections = n_states
    keep_connections = max(1, int(max_connections * (100 - sparsity_percent) / 100))
    
    print(f"🔬 Creating {sparsity_percent}% sparse transitions: {keep_connections}/{max_connections} connections per state")
    
    transmat = np.zeros((n_states, n_states))
    
    for state in range(n_states):
        # Randomly select which connections to keep for this state
        available_targets = list(range(n_states))
        np.random.shuffle(available_targets)
        selected_targets = available_targets[:keep_connections]
        
        # Generate Dirichlet probabilities only for selected targets
        probs = np.random.dirichlet(np.ones(keep_connections))
        
        # Assign probabilities to selected targets
        for i, target in enumerate(selected_targets):
            transmat[state, target] = probs[i]
    
    # Verify each row sums to 1
    row_sums = transmat.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Transition matrix rows don't sum to 1: {row_sums}"
    
    # Calculate actual sparsity
    total_entries = n_states * n_states
    zero_entries = np.sum(transmat == 0)
    actual_sparsity = (zero_entries / total_entries) * 100
    
    print(f"✅ Sparse transition matrix: {actual_sparsity:.1f}% zero entries ({zero_entries}/{total_entries})")
    
    return transmat

def create_mm_model(n_states, seed, sparsity=None):
    """Create MM model with specified parameters.
    
    CRITICAL WARNING: Do NOT set random_state=seed in the HMM constructor below!
    This was a critical bug that caused identical sequence generation, making
    models appear to learn perfectly (99%+ accuracy) while having terrible
    mechanistic interpretability (KL divergence worse than random baseline).
    
    The bug: random_state=seed forces identical sequence generation
    The fix: Let HMM use its own random state for diverse sequences
    """
    np.random.seed(seed)
    
    # Create MM with Dirichlet-sampled transitions
    # Note: Don't set random_state here to allow diverse sequence generation
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = np.random.dirichlet(np.ones(n_states))
    
    if sparsity is not None:
        # Create sparse transition matrix
        model.transmat_ = create_sparse_transitions(n_states, sparsity, seed)
    else:
        # Dense transition matrix (original behavior)
        model.transmat_ = np.random.dirichlet(np.ones(n_states), size=n_states)
    
    model.emissionprob_ = np.eye(n_states)  # Direct state->token mapping
    
    return model

def generate_sequences_from_model(model, n_states, n_sequences, use_bos=True, block_size=128):
    """Generate sequences from an existing MM model."""
    sequences = []
    
    # Fixed sequence length based on block_size - 1 to account for BOS
    seq_length = block_size - 1
    
    # Progress reporting
    report_interval = max(1, n_sequences // 20)  # Report 20 times or every sequence
    
    for i in range(n_sequences):
        if i % report_interval == 0:
            percent = (i / n_sequences) * 100
            print(f"  Generating sequences: {i:,}/{n_sequences:,} ({percent:.1f}%)", end='\r')
        
        seq, _ = model.sample(seq_length)
        tokens = seq.flatten().tolist()
        
        # Add BOS token (n_states) at the beginning if enabled
        if use_bos:
            tokens = [n_states] + tokens  # BOS token is vocab_size - 1
            
        sequences.append(tokens)
    
    print(f"  Generating sequences: {n_sequences:,}/{n_sequences:,} (100.0%) ✓")
    
    return sequences

def save_as_binary(sequences, filename):
    """Save sequences in nanoGPT binary format"""
    # Concatenate all sequences
    data = np.concatenate([np.array(s, dtype=np.uint16) for s in sequences])
    
    # Save as memmap
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(len(data),))
    arr[:] = data
    arr.flush()
    print(f"Saved {len(data):,} tokens to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_states', type=int, help='Number of MM states')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=int, default=10000, help='Training sequences')
    parser.add_argument('--val', type=int, default=1000, help='Validation sequences')
    parser.add_argument('--sparsity', type=int, help='Sparsity percentage (e.g., 25 for 25% sparse)')
    args = parser.parse_args()
    
    if args.sparsity:
        print(f"🎲 Generating MM-{args.n_states} SPARSE-{args.sparsity}% data with seed {args.seed}")
    else:
        print(f"🎲 Generating MM-{args.n_states} data with seed {args.seed}")
    
    # Vocab size includes BOS token
    vocab_size = args.n_states + 1  # States 0..n_states-1 + BOS token
    block_size = 256 if args.n_states > 10 else 128
    
    print(f"📊 Using BOS token: vocab_size={vocab_size}, BOS token={args.n_states}")
    print(f"📏 Fixed sequence length: {block_size} tokens (block-aligned)")
    
    # Create directory structure first
    if args.sparsity:
        data_dir = f'data/mm{args.n_states}-sparse-{args.sparsity}'
    else:
        data_dir = f'data/mm{args.n_states}'
    
    print(f"\n📁 Creating directory {data_dir}/...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create and save the model FIRST
    print(f"🎲 Creating MM-{args.n_states} model...")
    model = create_mm_model(args.n_states, args.seed, args.sparsity)
    
    # Save model immediately
    print(f"💾 Saving model.pkl...")
    with open(f'{data_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata (without sequence counts yet)
    print(f"💾 Saving initial metadata...")
    metadata = {
        'n_states': args.n_states,
        'vocab_size': vocab_size,
        'bos_token': args.n_states,
        'block_size': block_size,
        'seed': args.seed,
        'sparsity': args.sparsity,
        'status': 'model_created'
    }
    with open(f'{data_dir}/meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Now generate sequences
    print(f"\n📝 Generating training sequences...")
    train_seqs = generate_sequences_from_model(model, args.n_states, args.train, use_bos=True, block_size=block_size)
    
    print(f"\n📝 Generating validation sequences...")
    val_seqs = generate_sequences_from_model(model, args.n_states, args.val, use_bos=True, block_size=block_size)
    
    # Save sequences in binary format
    print(f"\n💾 Saving sequences...")
    save_as_binary(train_seqs, f'{data_dir}/train.bin')
    save_as_binary(val_seqs, f'{data_dir}/val.bin')
    
    # Update metadata with final information
    metadata['train_sequences'] = args.train
    metadata['val_sequences'] = args.val
    metadata['train_tokens'] = len(np.concatenate([np.array(s) for s in train_seqs]))
    metadata['val_tokens'] = len(np.concatenate([np.array(s) for s in val_seqs]))
    metadata['status'] = 'complete'
    
    # Add transition matrix sparsity analysis for sparse models
    if args.sparsity:
        total_entries = args.n_states * args.n_states
        zero_entries = np.sum(model.transmat_ == 0)
        actual_sparsity = (zero_entries / total_entries) * 100
        metadata['transition_sparsity'] = {
            'requested_percent': args.sparsity,
            'actual_percent': actual_sparsity,
            'zero_entries': int(zero_entries),
            'total_entries': total_entries
        }
    
    with open(f'{data_dir}/meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save meta.pkl for train_streamlined.py compatibility
    with open(f'{data_dir}/meta.pkl', 'wb') as f:
        pickle.dump({'vocab_size': vocab_size}, f)
    
    print(f"✅ Generated {metadata['train_tokens']:,} train tokens, {metadata['val_tokens']:,} val tokens")
    print(f"📁 Files in {data_dir}/ directory: train.bin, val.bin, model.pkl")
