#!/usr/bin/env python3
"""
MM-extra Metrics Aggregation Script

Aggregates all available metrics for MM-extra experiments including:
- Basic MM evaluation (perplexity, accuracy)
- Transition Matrix Fidelity (Metric 4)
- Markov Property Verification (Metric 5)
- Model architecture parameters

Usage:
    python MM/aggregate_metrics.py
    python MM/aggregate_metrics.py --output mm_comprehensive_metrics.csv
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def parse_model_eval_txt(file_path):
    """Parse model_eval.txt file to extract basic metrics"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract model config
        config_match = re.search(r'GPTConfig\((.*?)\)', content, re.DOTALL)
        config_str = config_match.group(1) if config_match else ""
        
        # Parse specific config values
        n_layer = int(re.search(r'n_layer=(\d+)', config_str).group(1)) if re.search(r'n_layer=(\d+)', config_str) else None
        n_head = int(re.search(r'n_head=(\d+)', config_str).group(1)) if re.search(r'n_head=(\d+)', config_str) else None
        n_embd = int(re.search(r'n_embd=(\d+)', config_str).group(1)) if re.search(r'n_embd=(\d+)', config_str) else None
        mlp_only = 'mlp_only=True' in config_str
        one_hot_embed = 'one_hot_embed=True' in config_str
        mlp_filter = re.search(r"mlp_filter='(\w+)'", config_str).group(1) if re.search(r"mlp_filter='(\w+)'", config_str) else "GELU"
        
        # Extract parameters
        params_match = re.search(r'number of parameters: ([\d.]+)M', content)
        parameters = float(params_match.group(1)) * 1e6 if params_match else None
        
        # Extract performance metrics (works for any seed)
        perf_match = re.search(r'Seed \d+: Perplexity=([\d.]+), Accuracy=([\d.]+)%', content)
        if perf_match:
            perplexity = float(perf_match.group(1))
            accuracy = float(perf_match.group(2))
        else:
            perplexity, accuracy = None, None
            
        return {
            'n_layer': n_layer,
            'n_head': n_head, 
            'n_embd': n_embd,
            'parameters': parameters,
            'mlp_only': mlp_only,
            'one_hot_embed': one_hot_embed,
            'mlp_filter': mlp_filter,
            'perplexity': perplexity,
            'accuracy_pct': accuracy
        }
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}

def load_mm_metrics_json(file_path):
    """Load transition matrix metrics from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            'metric4_kl_mean': data.get('mean_kl_divergence'),
            'metric4_kl_std': data.get('std_kl_divergence'),
            'metric4_performance': data.get('performance_level'),
            'metric4_success': data.get('is_successful'),
            'metric5_pairwise_kl_mean': data.get('markov_property_results', {}).get('mean_pairwise_kl'),
            'metric5_pairwise_kl_std': data.get('markov_property_results', {}).get('std_pairwise_kl'),
            'metric5_success': data.get('markov_property_results', {}).get('success'),
            'metric5_tokens_evaluated': data.get('markov_property_results', {}).get('tokens_evaluated')
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def get_variant_description(variant):
    """Get human-readable description for each variant"""
    descriptions = {
        'c1': '1L2H256D - Reduced layers + heads',
        'c2': '1L1H128D - Minimal viable transformer', 
        'c3': '1L1H101D - Frozen one-hot + attention',
        'c4': '1L1H101D - Learned embeddings + attention',
        'c5': '1L MLP-only + one-hot (RELU)',
        'c6': '1L MLP-only + one-hot (GELU, 1x ratio)',
        'c7': 'c4 + no positional embeddings',
        'c8': 'c4 + no LayerNorm',
        'c9': 'c4 + no LayerNorm + no pos embeddings',
        'c10': 'c9 + identity first matrix',
        'c11': 'c9 + MLP-only',
        'c12': 'c10 + MLP-only'
    }
    return descriptions.get(variant, f'Variant {variant}')

def main():
    parser = argparse.ArgumentParser(description='Aggregate MM-extra metrics')
    parser.add_argument('--output', type=str, default='MM/mm_comprehensive_metrics.csv',
                       help='Output CSV file path')
    parser.add_argument('--base_dir', type=str, default='trainings/MM/MM-extra',
                       help='Base directory for MM-extra experiments')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    all_metrics = []
    
    # Detect whether this is MM-extra (c1, c2, etc.) or base MM (seeds)
    extra_variants = list(base_dir.glob('*-c*-s42'))  # MM-extra pattern
    seed_variants = list(base_dir.glob('*-42')) + list(base_dir.glob('*-123')) + list(base_dir.glob('*-999'))  # Base MM pattern
    
    if extra_variants:
        # Process MM-extra variants
        variants = []
        for variant_dir in base_dir.glob('*-c*-s42'):
            # Extract variant from pattern like MM-100-sparse-75-extra-c1-s42
            parts = variant_dir.name.split('-')
            variant = parts[-2]  # Extract 'c1', 'c2', etc.
            variants.append(variant)
        
        variants = sorted(set(variants), key=lambda x: int(x[1:]))  # Sort by number
        print(f"📊 Aggregating metrics for {len(variants)} MM-extra variants...")
        print(f"Variants found: {variants}")
        
        for variant in variants:
            # Find the full directory name for this variant
            variant_dirs = list(base_dir.glob(f'*-{variant}-s42'))
            if not variant_dirs:
                continue
            variant_dir = variant_dirs[0]
            
            print(f"\n🔍 Processing {variant}...")
            
            # Initialize metrics dict
            metrics = {
                'variant': variant,
                'description': get_variant_description(variant)
            }
            
            # Load basic metrics from model_eval.txt
            model_eval_file = variant_dir / 'model_eval.txt'
            if model_eval_file.exists():
                basic_metrics = parse_model_eval_txt(model_eval_file)
                metrics.update(basic_metrics)
                print(f"  ✓ Basic metrics: {basic_metrics.get('accuracy_pct', 'N/A')}% accuracy, {basic_metrics.get('perplexity', 'N/A')} perplexity")
            else:
                print(f"  ❌ No model_eval.txt found")
            
            # Load transition matrix metrics from mm_metrics_4_5.json
            mm_metrics_file = variant_dir / 'mm_metrics_4_5.json'
            if mm_metrics_file.exists():
                mm_metrics = load_mm_metrics_json(mm_metrics_file)
                metrics.update(mm_metrics)
                print(f"  ✓ MM metrics: Metric 4 KL={mm_metrics.get('metric4_kl_mean', 'N/A'):.6f}, Metric 5 success={mm_metrics.get('metric5_success', 'N/A')}")
            else:
                print(f"  ⚠️  No mm_metrics_4_5.json found")
            
            all_metrics.append(metrics)
            
    elif seed_variants:
        # Process base MM models with seeds
        seeds = ['42', '123', '999']
        print(f"📊 Aggregating metrics for base MM model with {len(seeds)} seeds...")
        
        for seed in seeds:
            # Find seed directory
            seed_dirs = list(base_dir.glob(f'*-{seed}'))
            if not seed_dirs:
                continue
            seed_dir = seed_dirs[0]
            
            print(f"\n🔍 Processing seed {seed}...")
            
            # Initialize metrics dict
            metrics = {
                'variant': f'seed-{seed}',
                'description': f'Base model with seed {seed}'
            }
            
            # Load basic metrics from model_eval.txt
            model_eval_file = seed_dir / 'model_eval.txt'
            if model_eval_file.exists():
                basic_metrics = parse_model_eval_txt(model_eval_file)
                metrics.update(basic_metrics)
                print(f"  ✓ Basic metrics: {basic_metrics.get('accuracy_pct', 'N/A')}% accuracy, {basic_metrics.get('perplexity', 'N/A')} perplexity")
            else:
                print(f"  ❌ No model_eval.txt found")
            
            # Load transition matrix metrics from mm_metrics_4_5.json
            mm_metrics_file = seed_dir / 'mm_metrics_4_5.json'
            if mm_metrics_file.exists():
                mm_metrics = load_mm_metrics_json(mm_metrics_file)
                metrics.update(mm_metrics)
                print(f"  ✓ MM metrics: Metric 4 KL={mm_metrics.get('metric4_kl_mean', 'N/A'):.6f}, Metric 5 success={mm_metrics.get('metric5_success', 'N/A')}")
            else:
                print(f"  ⚠️  No mm_metrics_4_5.json found")
            
            all_metrics.append(metrics)
    else:
        print("❌ No valid MM models found in base directory")
        return pd.DataFrame()
    
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for better readability
    column_order = [
        'variant', 'description',
        'n_layer', 'n_head', 'n_embd', 'parameters',
        'mlp_only', 'one_hot_embed', 'mlp_filter',
        'perplexity', 'accuracy_pct',
        'metric4_kl_mean', 'metric4_kl_std', 'metric4_performance', 'metric4_success',
        'metric5_pairwise_kl_mean', 'metric5_pairwise_kl_std', 'metric5_success', 'metric5_tokens_evaluated'
    ]
    
    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\n📈 Comprehensive metrics table saved to: {output_path}")
    print(f"📊 Total variants processed: {len(df)}")
    
    # Print summary statistics
    print(f"\n📋 Summary Statistics:")
    print(f"{'Variant':<12} {'Params':<8} {'Accuracy':<10} {'Perplexity':<12} {'Metric4 KL':<12} {'Metric5 Success':<15}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        variant = row['variant']
        params = f"{row.get('parameters', 0)/1e6:.2f}M" if pd.notna(row.get('parameters')) else 'N/A'
        accuracy = f"{row.get('accuracy_pct', 0):.2f}%" if pd.notna(row.get('accuracy_pct')) else 'N/A'
        perplexity = f"{row.get('perplexity', 0):.2f}" if pd.notna(row.get('perplexity')) else 'N/A'
        metric4_kl = f"{row.get('metric4_kl_mean', 0):.6f}" if pd.notna(row.get('metric4_kl_mean')) else 'N/A'
        metric5_success = str(row.get('metric5_success', 'N/A'))
        
        print(f"{variant:<12} {params:<8} {accuracy:<10} {perplexity:<12} {metric4_kl:<12} {metric5_success:<15}")
    
    # Architectural insights (only for MM-extra variants that have mlp_only column)
    print(f"\n🔬 Key Architectural Insights:")
    if 'mlp_only' in df.columns and len(df) > 0:
        attention_models = df[df['mlp_only'] == False]
        mlp_models = df[df['mlp_only'] == True]
        
        if len(attention_models) > 0 and len(mlp_models) > 0:
            att_acc_mean = attention_models['accuracy_pct'].mean() if 'accuracy_pct' in attention_models.columns else 0
            mlp_acc_mean = mlp_models['accuracy_pct'].mean() if 'accuracy_pct' in mlp_models.columns else 0
            print(f"  • Attention models average accuracy: {att_acc_mean:.2f}%")
            print(f"  • MLP-only models average accuracy: {mlp_acc_mean:.2f}%")
            
            if 'metric4_kl_mean' in df.columns:
                att_metric4_mean = attention_models['metric4_kl_mean'].mean()
                mlp_metric4_mean = mlp_models['metric4_kl_mean'].mean()
                print(f"  • Attention models average Metric 4 KL: {att_metric4_mean:.6f}")
                print(f"  • MLP-only models average Metric 4 KL: {mlp_metric4_mean:.6f}")
    else:
        print(f"  • Architectural analysis not available (no mlp_only data or empty dataset)")
    
    # Parameter efficiency
    if 'parameters' in df.columns and 'accuracy_pct' in df.columns and len(df) > 0:
        efficiency = df['accuracy_pct'] / (df['parameters'] / 1e6)  # accuracy per million parameters
        if len(efficiency) > 0 and not efficiency.isna().all():
            best_efficiency_idx = efficiency.idxmax()
            best_variant = df.loc[best_efficiency_idx, 'variant']
            best_efficiency_val = efficiency.iloc[best_efficiency_idx]
            print(f"  • Most parameter-efficient: {best_variant} ({best_efficiency_val:.2f} acc%/M params)")
        else:
            print(f"  • Parameter efficiency analysis not available (no valid data)")
    else:
        print(f"  • Parameter efficiency analysis not available (missing columns or empty dataset)")
    
    return df

if __name__ == "__main__":
    df = main()