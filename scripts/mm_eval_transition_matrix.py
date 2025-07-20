#!/usr/bin/env python3
"""
MM-extra Transition Matrix Fidelity Evaluation (Metric 4)

This script implements the novel transition matrix fidelity metric that directly
compares a transformer's learned transition probabilities with the formal MM 
transition matrix.

Usage:
    python mm_eval_transition_matrix.py --transformer_path <path> --formal_model_path <path>
    
Examples:
    python mm_eval_transition_matrix.py \
        --transformer_path trainings/MM/MM-extra/c3-s42/ckpt.pt \
        --formal_model_path data/delaylang/MM/mm100/model.pkl
        
    python mm_eval_transition_matrix.py \
        --transformer_path trainings/MM/MM-extra/c8-s42/ckpt.pt \
        --formal_model_path data/delaylang/MM/mm100/model.pkl \
        --output results/c8_transition_eval.json
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from model import GPT, GPTConfig

def load_ground_truth_transitions(formal_model_path):
    """Load the original MM transition matrix from hmmlearn model"""
    formal_model_path = Path(formal_model_path)
    
    if not formal_model_path.exists():
        raise FileNotFoundError(f"Ground truth MM model not found: {formal_model_path}")
    
    with open(formal_model_path, 'rb') as f:
        mm_model = pickle.load(f)
    
    # Extract transition matrix
    # Note: hmmlearn uses transmat_ attribute for transition probabilities
    transition_matrix = mm_model.transmat_
    vocab_size = transition_matrix.shape[0]
    
    print(f"Loaded MM transition matrix from: {formal_model_path}")
    print(f"Transition matrix shape: {transition_matrix.shape}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Verify it's a valid transition matrix
    row_sums = np.sum(transition_matrix, axis=1)
    if not np.allclose(row_sums, 1.0):
        print(f"Warning: Transition matrix rows don't sum to 1.0: {row_sums}")
    
    return transition_matrix, vocab_size

def load_transformer_model(transformer_path):
    """Load trained transformer model from checkpoint path"""
    transformer_path = Path(transformer_path)
    
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {transformer_path}")
    
    print(f"Loading transformer from: {transformer_path}")
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(transformer_path, map_location=device)
    
    # Reconstruct model config from checkpoint
    if 'model_args' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'model_args': {transformer_path}")
    
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load model weights
    state_dict = checkpoint['model']
    # Handle potential prefix issues
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Model vocab_size: {gptconf.vocab_size}")
    print(f"Model config: {gptconf}")
    
    return model, device, gptconf

def analyze_baselines(transition_matrix, vocab_size):
    """Calculate KL divergence baselines for calibration"""
    print(f"\n=== Baseline KL Divergence Analysis ===")
    
    all_uniform_kls = []
    all_random_kls = []
    all_noisy_kls = []
    
    # Sample a few tokens for detailed examples
    example_tokens = [0, 1, vocab_size//4, vocab_size//2, vocab_size-1]
    examples = []
    
    for token in range(vocab_size):
        ground_truth = transition_matrix[token, :]
        
        # Baseline 1: Uniform distribution
        uniform_probs = np.ones(vocab_size) / vocab_size
        kl_uniform = np.sum(ground_truth * np.log(np.clip(ground_truth, 1e-8, 1.0) / np.clip(uniform_probs, 1e-8, 1.0)))
        
        # Baseline 2: Random distribution
        np.random.seed(42 + token)  # Reproducible
        random_probs = np.random.dirichlet(np.ones(vocab_size))
        kl_random = np.sum(ground_truth * np.log(np.clip(ground_truth, 1e-8, 1.0) / np.clip(random_probs, 1e-8, 1.0)))
        
        # Baseline 3: Slightly noisy ground truth
        noise_level = 0.01
        np.random.seed(42 + token)
        noise = np.random.normal(0, noise_level, vocab_size)
        noisy_probs = ground_truth + noise
        noisy_probs = np.clip(noisy_probs, 1e-8, 1.0)
        noisy_probs = noisy_probs / np.sum(noisy_probs)  # Renormalize
        kl_noisy = np.sum(ground_truth * np.log(np.clip(ground_truth, 1e-8, 1.0) / np.clip(noisy_probs, 1e-8, 1.0)))
        
        all_uniform_kls.append(kl_uniform)
        all_random_kls.append(kl_random)
        all_noisy_kls.append(kl_noisy)
        
        # Collect examples for detailed output
        if token in example_tokens:
            examples.append({
                'token': int(token),
                'ground_truth_top3': [(int(idx), float(prob)) for idx, prob in sorted(enumerate(ground_truth), key=lambda x: x[1], reverse=True)[:3]],
                'uniform_top3': [(int(idx), float(prob)) for idx, prob in sorted(enumerate(uniform_probs), key=lambda x: x[1], reverse=True)[:3]],
                'random_top3': [(int(idx), float(prob)) for idx, prob in sorted(enumerate(random_probs), key=lambda x: x[1], reverse=True)[:3]],
                'kl_uniform': float(kl_uniform),
                'kl_random': float(kl_random),
                'kl_noisy': float(kl_noisy)
            })
    
    baseline_results = {
        'uniform': {'mean': float(np.mean(all_uniform_kls)), 'std': float(np.std(all_uniform_kls)), 'min': float(np.min(all_uniform_kls)), 'max': float(np.max(all_uniform_kls))},
        'random': {'mean': float(np.mean(all_random_kls)), 'std': float(np.std(all_random_kls)), 'min': float(np.min(all_random_kls)), 'max': float(np.max(all_random_kls))},
        'noisy': {'mean': float(np.mean(all_noisy_kls)), 'std': float(np.std(all_noisy_kls)), 'min': float(np.min(all_noisy_kls)), 'max': float(np.max(all_noisy_kls))}
    }
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"\n1. UNIFORM BASELINE (all tokens equally likely):")
    print(f"   Mean KL: {baseline_results['uniform']['mean']:.6f} ± {baseline_results['uniform']['std']:.6f}")
    print(f"   Range:   {baseline_results['uniform']['min']:.6f} - {baseline_results['uniform']['max']:.6f}")
    
    print(f"\n2. RANDOM BASELINE (random probability distribution):")
    print(f"   Mean KL: {baseline_results['random']['mean']:.6f} ± {baseline_results['random']['std']:.6f}")
    print(f"   Range:   {baseline_results['random']['min']:.6f} - {baseline_results['random']['max']:.6f}")
    
    print(f"\n3. NOISY BASELINE (ground truth + 1% noise):")
    print(f"   Mean KL: {baseline_results['noisy']['mean']:.6f} ± {baseline_results['noisy']['std']:.6f}")
    print(f"   Range:   {baseline_results['noisy']['min']:.6f} - {baseline_results['noisy']['max']:.6f}")
    
    print(f"\n=== Human-Readable Examples ===")
    for example in examples:
        token = example['token']
        print(f"\nToken {token}:")
        print(f"  Ground Truth - Top 3 transitions: {[(f'→{idx}', f'{prob:.4f}') for idx, prob in example['ground_truth_top3']]}")
        print(f"  Uniform      - All transitions:   [('→any', {1.0/vocab_size:.4f})]")
        print(f"  Random       - Top 3 transitions: {[(f'→{idx}', f'{prob:.4f}') for idx, prob in example['random_top3']]}")
        print(f"  KL Divergences: Uniform={example['kl_uniform']:.4f}, Random={example['kl_random']:.4f}, Noisy={example['kl_noisy']:.4f}")
    
    return baseline_results, examples

def compute_transition_fidelity(model, device, config, transition_matrix, vocab_size, formal_model_path):
    """
    Compute transition matrix fidelity metric with multiple diagnostic approaches.
    
    Tests different prompt formats to diagnose why model has good perplexity but poor transition fidelity.
    """
    print(f"\nComputing transition matrix fidelity - DIAGNOSTIC MODE...")
    
    # Verify vocab sizes match
    model_vocab_size = config.vocab_size
    if vocab_size + 1 != model_vocab_size:  # +1 for BOS token
        print(f"Warning: Vocab size mismatch. MM vocab={vocab_size}, Model vocab={model_vocab_size}")
        print(f"Assuming BOS token is at index {vocab_size}")
    
    # BOS token is typically the last token in vocabulary
    bos_token = vocab_size
    
    # Infer validation data path from formal model path
    formal_model_path = Path(formal_model_path)
    val_data_path = formal_model_path.parent / 'val.bin'
    if os.path.exists(val_data_path):
        val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
        print(f"Loaded validation data: {len(val_data)} tokens")
    else:
        val_data = None
        print("Warning: No validation data found, using synthetic prompts only")
    
    diagnostics = {}
    
    # Test 1: Original BOS + token approach
    print(f"\n=== TEST 1: BOS + Token Approach ===")
    test1_results = run_transition_test(model, device, transition_matrix, vocab_size, bos_token, "bos_token", val_data)
    diagnostics['bos_token'] = test1_results
    
    # Test 2: No BOS, just single token
    print(f"\n=== TEST 2: Single Token (No BOS) ===")
    test2_results = run_transition_test(model, device, transition_matrix, vocab_size, None, "no_bos", val_data)
    diagnostics['no_bos'] = test2_results
    
    # Test 3: Random 4-token prefix + target token
    print(f"\n=== TEST 3: Random 4-Token Prefix ===")
    test3_results = run_transition_test(model, device, transition_matrix, vocab_size, None, "random_prefix", val_data)
    diagnostics['random_prefix'] = test3_results
    
    # Test 4: Real validation sequences at different positions
    if val_data is not None:
        print(f"\n=== TEST 4: Real Validation Sequences ===")
        test4_results = run_validation_sequence_test(model, device, transition_matrix, vocab_size, val_data)
        diagnostics['validation_sequences'] = test4_results
    
    # Use original BOS approach as main result for compatibility
    main_results = test1_results
    
    return main_results['mean_kl'], main_results['std_kl'], main_results['min_kl'], main_results['max_kl'], main_results['per_token_results'], diagnostics

def run_transition_test(model, device, transition_matrix, vocab_size, bos_token, test_name, val_data):
    """Run a single transition fidelity test with specific prompt format"""
    kl_divergences = []
    per_token_results = {}
    
    with torch.no_grad():
        for current_token in range(vocab_size):
            # Create different prompt types
            if test_name == "bos_token":
                prompt = torch.tensor([[bos_token, current_token]], dtype=torch.long, device=device)
            elif test_name == "no_bos":
                prompt = torch.tensor([[current_token]], dtype=torch.long, device=device)
            elif test_name == "random_prefix":
                # Generate random 4-token prefix + current token
                np.random.seed(42 + current_token)  # Reproducible
                prefix = np.random.choice(vocab_size, 4).tolist()
                prompt = torch.tensor([prefix + [current_token]], dtype=torch.long, device=device)
            
            # Forward pass
            logits, _ = model(prompt)
            
            # Extract next-token logits (after current_token)
            next_token_logits = logits[0, -1, :vocab_size]
            
            # Convert to probabilities
            model_probs = F.softmax(next_token_logits, dim=0).cpu().numpy()
            
            # Ground truth probabilities
            ground_truth = transition_matrix[current_token, :]
            
            # Compute KL divergence
            epsilon = 1e-8
            model_probs_safe = np.clip(model_probs, epsilon, 1.0)
            ground_truth_safe = np.clip(ground_truth, epsilon, 1.0)
            
            kl_div = np.sum(ground_truth_safe * np.log(ground_truth_safe / model_probs_safe))
            kl_divergences.append(kl_div)
            
            # Store detailed results
            per_token_results[current_token] = {
                'kl_divergence': float(kl_div),
                'ground_truth': [float(x) for x in ground_truth.tolist()],
                'model_probs': [float(x) for x in model_probs.tolist()],
                'prompt': prompt[0].cpu().numpy().tolist(),
                'ground_truth_top3': [(int(idx), float(prob)) for idx, prob in sorted(enumerate(ground_truth), key=lambda x: x[1], reverse=True)[:3]],
                'model_probs_top3': [(int(idx), float(prob)) for idx, prob in sorted(enumerate(model_probs), key=lambda x: x[1], reverse=True)[:3]]
            }
            
            # Progress reporting for first few tokens
            if current_token < 5:
                print(f"  Token {current_token:3d}: KL={kl_div:.6f}, Prompt={prompt[0].cpu().numpy().tolist()}")
    
    mean_kl = np.mean(kl_divergences)
    print(f"  {test_name.upper()} Mean KL: {mean_kl:.6f}")
    
    return {
        'mean_kl': float(mean_kl),
        'std_kl': float(np.std(kl_divergences)),
        'min_kl': float(np.min(kl_divergences)),
        'max_kl': float(np.max(kl_divergences)),
        'per_token_results': per_token_results
    }

def run_validation_sequence_test(model, device, transition_matrix, vocab_size, val_data):
    """Test on real validation sequences at different positions"""
    print("  Testing transition agreement at positions 0, 10, 20, 30...")
    
    # Sample some validation sequences
    seq_length = 32
    n_sequences = 50
    positions_to_test = [0, 10, 20, 30]
    
    position_results = {}
    
    with torch.no_grad():
        for pos in positions_to_test:
            if pos >= seq_length - 1:
                continue
                
            kl_divergences = []
            agreements = []
            
            for seq_idx in range(n_sequences):
                # Get a random sequence from validation data
                start_idx = np.random.randint(0, len(val_data) - seq_length)
                sequence = val_data[start_idx:start_idx + seq_length]
                
                # Extract current token at position
                if pos >= len(sequence) - 1:
                    continue
                    
                current_token = int(sequence[pos])
                if current_token >= vocab_size:  # Skip BOS tokens
                    continue
                
                # Create prompt up to position
                prompt_tokens = sequence[:pos+1].astype(np.int64)
                prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                
                # Forward pass
                logits, _ = model(prompt)
                next_token_logits = logits[0, -1, :vocab_size]
                model_probs = F.softmax(next_token_logits, dim=0).cpu().numpy()
                
                # Ground truth for this token
                ground_truth = transition_matrix[current_token, :]
                
                # Compute KL divergence
                epsilon = 1e-8
                model_probs_safe = np.clip(model_probs, epsilon, 1.0)
                ground_truth_safe = np.clip(ground_truth, epsilon, 1.0)
                kl_div = np.sum(ground_truth_safe * np.log(ground_truth_safe / model_probs_safe))
                kl_divergences.append(kl_div)
                
                # Check if actual next token matches top prediction
                if pos + 1 < len(sequence):
                    actual_next = int(sequence[pos + 1])
                    if actual_next < vocab_size:
                        predicted_next = np.argmax(model_probs)
                        agreements.append(1 if predicted_next == actual_next else 0)
            
            if kl_divergences:
                position_results[pos] = {
                    'mean_kl': float(np.mean(kl_divergences)),
                    'accuracy': float(np.mean(agreements)) if agreements else 0.0,
                    'n_samples': len(kl_divergences)
                }
                print(f"    Position {pos:2d}: KL={position_results[pos]['mean_kl']:.4f}, Acc={position_results[pos]['accuracy']:.3f}, N={position_results[pos]['n_samples']}")
    
    return position_results

def evaluate_performance(mean_kl, baseline_results):
    """Evaluate model performance against baselines"""
    uniform_kl = baseline_results['uniform']['mean']
    random_kl = baseline_results['random']['mean']
    noisy_kl = baseline_results['noisy']['mean']
    
    print(f"\n=== Performance Evaluation ===")
    print(f"Model KL:   {mean_kl:.6f}")
    print(f"Uniform KL: {uniform_kl:.6f}")
    print(f"Random KL:  {random_kl:.6f}")
    print(f"Noisy KL:   {noisy_kl:.6f}")
    
    # Determine performance level
    if mean_kl < noisy_kl * 2:
        performance = "EXCELLENT"
        success = True
        print(f"✅ EXCELLENT: Model KL < 2× Noisy baseline ({noisy_kl * 2:.6f})")
    elif mean_kl < uniform_kl * 0.5:
        performance = "GOOD"
        success = True
        print(f"✅ GOOD: Model KL < 0.5× Uniform baseline ({uniform_kl * 0.5:.6f})")
    elif mean_kl < uniform_kl:
        performance = "FAIR"
        success = True
        print(f"🟡 FAIR: Model KL < Uniform baseline ({uniform_kl:.6f})")
    elif mean_kl < random_kl:
        performance = "POOR"
        success = False
        print(f"🟠 POOR: Model KL < Random baseline ({random_kl:.6f})")
    else:
        performance = "VERY_POOR"
        success = False
        print(f"❌ VERY POOR: Model KL > Random baseline ({random_kl:.6f})")
    
    return performance, success

def print_detailed_examples(per_token_results, example_tokens=[0, 1, 2]):
    """Print detailed comparison of ground truth vs model for specific tokens"""
    print(f"\n=== Detailed Token Analysis ===")
    
    for token in example_tokens:
        if token not in per_token_results:
            continue
            
        data = per_token_results[token]
        gt_top3 = data['ground_truth_top3']
        model_top3 = data['model_probs_top3']
        kl_div = data['kl_divergence']
        
        print(f"\nToken {token} → Next Token (KL Divergence: {kl_div:.6f}):")
        print(f"{'='*60}")
        
        # Collect all unique tokens mentioned in either top 3
        all_tokens = set()
        for idx, _ in gt_top3:
            all_tokens.add(idx)
        for idx, _ in model_top3:
            all_tokens.add(idx)
        
        print(f"{'Next Token':<12} {'Ground Truth':<15} {'Model':<15} {'Ratio (M/GT)':<12}")
        print(f"{'-'*60}")
        
        # Show ground truth top 3 first
        print("GROUND TRUTH TOP 3:")
        for idx, gt_prob in gt_top3:
            model_prob = data['model_probs'][idx]
            ratio = model_prob / gt_prob if gt_prob > 0 else float('inf')
            print(f"  →{idx:<9} {gt_prob:<15.6f} {model_prob:<15.6f} {ratio:<12.2e}")
        
        print("MODEL TOP 3:")
        for idx, model_prob in model_top3:
            gt_prob = data['ground_truth'][idx]
            ratio = model_prob / gt_prob if gt_prob > 0 else float('inf')
            print(f"  →{idx:<9} {gt_prob:<15.6f} {model_prob:<15.6f} {ratio:<12.2e}")
        
        # Check if there's any overlap
        gt_tokens = {idx for idx, _ in gt_top3}
        model_tokens = {idx for idx, _ in model_top3}
        overlap = gt_tokens.intersection(model_tokens)
        
        print(f"\nOverlap in top 3: {len(overlap)}/3 tokens {list(overlap) if overlap else '(none)'}")
        
        if overlap:
            print("OVERLAPPING TOKENS:")
            for idx in sorted(overlap):
                gt_prob = data['ground_truth'][idx]
                model_prob = data['model_probs'][idx]
                ratio = model_prob / gt_prob
                print(f"  →{idx:<9} {gt_prob:<15.6f} {model_prob:<15.6f} {ratio:<12.2e}")
    
    return True


# ===== METRIC 5: MARKOV PROPERTY VERIFICATION =====

def collect_token_contexts(data_path, vocab_size, min_contexts_per_token=10, max_history_length=32, max_samples=10000):
    """
    Collect diverse contexts for each token from validation data
    
    Returns: dict mapping token -> list of context arrays
    """
    
    print(f"\\nCollecting contexts from {data_path}...")
    print(f"Parameters: min_contexts={min_contexts_per_token}, max_history={max_history_length}")
    
    # Load validation data
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"Data size: {len(data):,} tokens")
    
    # Track contexts for each token (exclude BOS token)
    from collections import defaultdict
    token_contexts = defaultdict(list)
    n_states = vocab_size - 1  # Assume last token is BOS
    
    # Sample positions to avoid processing entire dataset
    np.random.seed(42)  # For reproducibility
    sample_positions = np.random.choice(
        range(max_history_length, len(data) - 1), 
        size=min(max_samples, len(data) - max_history_length - 1),
        replace=False
    )
    
    for i in sample_positions:
        current_token = int(data[i])
        
        # Skip BOS tokens and invalid tokens
        if current_token >= n_states:
            continue
            
        # Get history (prefix)
        start_idx = max(0, i - max_history_length)
        prefix = data[start_idx:i]
        
        # Store context if it's reasonable length
        if len(prefix) >= 8:  # Minimum context length
            token_contexts[current_token].append(prefix.copy())
        
        # Early termination if we have enough contexts
        if len(token_contexts) >= min(n_states // 2, 20):  # Check at least top 20 tokens
            sufficient_tokens = sum(1 for contexts in token_contexts.values() 
                                  if len(contexts) >= min_contexts_per_token)
            if sufficient_tokens >= min(n_states // 4, 10):  # Need at least 10 tokens with enough contexts
                break
    
    # Filter tokens with sufficient contexts
    valid_tokens = {}
    for token, contexts in token_contexts.items():
        if len(contexts) >= min_contexts_per_token:
            # Randomly sample to avoid bias and memory issues
            selected_indices = np.random.choice(
                len(contexts), 
                size=min_contexts_per_token, 
                replace=False
            )
            valid_tokens[token] = [contexts[i] for i in selected_indices]
    
    print(f"Collected contexts for {len(valid_tokens)} tokens (needed ≥{min_contexts_per_token} each)")
    
    if len(valid_tokens) == 0:
        raise ValueError("No tokens with sufficient contexts found!")
    
    return valid_tokens


def evaluate_markov_property(model, device, token_contexts, vocab_size):
    """
    Evaluate Markov property for each token (Metric 5)
    
    For each token T:
    1. Evaluate model on different contexts ending with T
    2. Calculate pairwise KL divergences between next-token distributions
    3. Mean pairwise KL indicates violation of Markov property
    
    Success Criterion: Mean pairwise KL < 0.1
    """
    
    print(f"\\n=== METRIC 5: Markov Property Verification ===")
    print(f"Tokens to evaluate: {len(token_contexts)}")
    
    n_states = vocab_size - 1  # Exclude BOS token from predictions
    token_violations = []
    
    with torch.no_grad():
        for token, contexts in token_contexts.items():
            
            context_probs = []
            
            for prefix in contexts:
                # Create sequence: prefix + token
                prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device)
                token_tensor = torch.tensor([token], dtype=torch.long, device=device)
                full_seq = torch.cat([prefix_tensor, token_tensor]).unsqueeze(0)
                
                # Limit sequence length to model's block size
                if full_seq.size(1) > model.config.block_size:
                    full_seq = full_seq[:, -model.config.block_size:]
                
                # Forward pass
                try:
                    logits, _ = model(full_seq)
                    
                    # Extract next-token probabilities (exclude BOS from predictions)
                    next_token_logits = logits[0, -1, :n_states]
                    probs = F.softmax(next_token_logits, dim=-1).cpu().numpy()
                    context_probs.append(probs)
                    
                except Exception as e:
                    print(f"Error processing token {token}: {e}")
                    continue
            
            if len(context_probs) < 2:
                print(f"Skipping token {token}: insufficient valid contexts")
                continue
            
            # Calculate pairwise KL divergences
            pairwise_kls = []
            epsilon = 1e-10
            
            for i in range(len(context_probs)):
                for j in range(i + 1, len(context_probs)):
                    prob_i = np.clip(context_probs[i], epsilon, 1.0)
                    prob_j = np.clip(context_probs[j], epsilon, 1.0)
                    
                    # Symmetric KL divergence
                    from scipy.stats import entropy
                    kl_ij = entropy(prob_i, prob_j)
                    kl_ji = entropy(prob_j, prob_i)
                    symmetric_kl = (kl_ij + kl_ji) / 2
                    
                    if not np.isnan(symmetric_kl) and not np.isinf(symmetric_kl):
                        pairwise_kls.append(symmetric_kl)
            
            if pairwise_kls:
                token_violation = np.mean(pairwise_kls)
                token_violations.append(token_violation)
                
                if len(token_violations) <= 5:  # Show first few for debugging
                    print(f"  Token {token}: {len(contexts)} contexts, {len(pairwise_kls)} pairs, KL={token_violation:.6f}")
    
    if not token_violations:
        raise ValueError("No valid token violations calculated!")
    
    # Calculate statistics
    mean_violation = np.mean(token_violations)
    std_violation = np.std(token_violations)
    max_violation = np.max(token_violations)
    min_violation = np.min(token_violations)
    
    success = mean_violation < 0.1
    
    print(f"\\n=== Markov Property Results ===")
    print(f"Tokens evaluated: {len(token_violations)}")
    print(f"Mean pairwise KL: {mean_violation:.6f}")
    print(f"Std pairwise KL:  {std_violation:.6f}")
    print(f"Min pairwise KL:  {min_violation:.6f}")
    print(f"Max pairwise KL:  {max_violation:.6f}")
    print(f"Success criterion: < 0.1 -> {'✅ PASS' if success else '❌ FAIL'}")
    
    return {
        'mean_pairwise_kl': float(mean_violation),
        'std_pairwise_kl': float(std_violation),
        'min_pairwise_kl': float(min_violation),
        'max_pairwise_kl': float(max_violation),
        'tokens_evaluated': len(token_violations),
        'individual_violations': [float(x) for x in token_violations],
        'success': bool(success),
        'success_threshold': 0.1
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate MM-extra transition matrix fidelity')
    parser.add_argument('--transformer_path', type=str, required=True, 
                       help='Path to transformer checkpoint (.pt file)')
    parser.add_argument('--formal_model_path', type=str, required=True,
                       help='Path to formal MM model (.pkl file)')
    parser.add_argument('--output', type=str, 
                       help='Output JSON file path (default: transition_matrix_eval.json)')
    
    args = parser.parse_args()
    
    try:
        print(f"=== MM-extra Transition Matrix Fidelity Evaluation ===")
        print(f"Transformer: {args.transformer_path}")
        print(f"Formal Model: {args.formal_model_path}")
        print()
        
        # Load ground truth MM transition matrix
        transition_matrix, vocab_size = load_ground_truth_transitions(args.formal_model_path)
        
        # Analyze baselines for calibration
        baseline_results, baseline_examples = analyze_baselines(transition_matrix, vocab_size)
        
        # Load trained transformer model
        model, device, config = load_transformer_model(args.transformer_path)
        
        # Compute transition matrix fidelity
        mean_kl, std_kl, min_kl, max_kl, per_token_results, diagnostics = compute_transition_fidelity(model, device, config, transition_matrix, vocab_size, args.formal_model_path)
        
        # Evaluate performance against baselines
        performance, success = evaluate_performance(mean_kl, baseline_results)
        
        # Print detailed examples
        print_detailed_examples(per_token_results, example_tokens=[0, 1, 2])
        
        # METRIC 5: Markov Property Verification
        try:
            # Determine validation data path from formal model path
            formal_model_path = Path(args.formal_model_path)
            val_data_path = formal_model_path.parent / 'val.bin'
            
            if val_data_path.exists():
                print(f"\\n{'='*60}")
                print(f"RUNNING METRIC 5: Markov Property Verification")
                print(f"{'='*60}")
                
                # Collect token contexts from validation data
                token_contexts = collect_token_contexts(val_data_path, vocab_size)
                
                # Evaluate Markov property
                markov_results = evaluate_markov_property(model, device, token_contexts, vocab_size)
                
            else:
                print(f"\\n❌ Skipping Metric 5: Validation data not found at {val_data_path}")
                markov_results = {'error': f'Validation data not found: {val_data_path}'}
                
        except Exception as e:
            print(f"\\n❌ Metric 5 failed: {e}")
            markov_results = {'error': str(e)}
        
        # Compile results
        results = {
            'mean_kl_divergence': float(mean_kl),
            'std_kl_divergence': float(std_kl),
            'min_kl_divergence': float(min_kl),
            'max_kl_divergence': float(max_kl),
            'performance_level': performance,
            'is_successful': success,
            'baseline_results': baseline_results,
            'baseline_examples': baseline_examples,
            'per_token_results': per_token_results,
            'diagnostics': diagnostics,
            'vocab_size': int(vocab_size),
            'markov_property_results': markov_results
        }
        
        # Add metadata
        results.update({
            'transformer_path': str(args.transformer_path),
            'formal_model_path': str(args.formal_model_path),
            'evaluation_metrics': ['transition_matrix_fidelity', 'markov_property_verification'],
            'description': 'Metric 4: Direct comparison of transformer transition probabilities with formal MM. Metric 5: Markov property verification via context independence testing.'
        })
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path("transition_matrix_eval.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Return exit code based on success
        sys.exit(0 if results['is_successful'] else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()