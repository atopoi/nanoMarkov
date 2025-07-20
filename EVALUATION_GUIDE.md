# Evaluation Guide

This guide explains the comprehensive evaluation framework for nanoMarkov models, including all metrics and how to run evaluations.

## Overview

The evaluation framework uses a 6-metric system to rigorously assess how well transformers learn Markov Model structures:

1. **Cross-Entropy Loss** - Direct optimization objective
2. **Perplexity** - Model uncertainty measure
3. **Accuracy** - Next-token prediction accuracy
4. **Agreement** - Inter-seed consistency check
5. **Transition Matrix Fidelity** - KL divergence from ground truth
6. **Markov Property Verification** - Tests true memorylessness

## Evaluation Scripts

### 1. Basic Evaluation: `scripts/mm_eval.py`

Computes fundamental metrics (1-3) and inter-model agreement.

```bash
# Evaluate a single model
python scripts/mm_eval.py --ckpt_path trainings/MM/MM-100/MM-100-42/ckpt.pt

# Evaluate multiple seeds and generate summary
python scripts/mm_eval.py 100 --seeds 42 123 999 --summary

# Evaluate sparse models
python scripts/mm_eval.py 100-sparse-75 --seeds 42

# Evaluate MM-extra variants
python scripts/mm_eval.py c12 --seeds 42
```

**Output**: 
- Perplexity (should be close to theoretical minimum)
- Accuracy (should beat random baseline by significant margin)
- Top-1 agreement between models (>80% indicates consistency)
- JS divergence between model predictions

### 2. Advanced Metrics: `scripts/mm_eval_transition_matrix.py`

Implements Metrics 4 & 5 - the most sophisticated evaluation metrics.

```bash
# Evaluate transition matrix fidelity
python scripts/mm_eval_transition_matrix.py \
    --transformer_path trainings/MM/MM-100/MM-100-42/ckpt.pt \
    --formal_model_path data/mm100/model.pkl \
    --output trainings/MM/MM-100/MM-100-42/mm_metrics_4_5.json
```

**Metric 4 - Transition Matrix Fidelity:**
- Directly compares learned vs ground truth transition probabilities
- Tests multiple prompt formats to diagnose model behavior
- Performance levels: EXCELLENT (<0.01), GOOD (<0.1), FAIR (<0.5), POOR (<1.0)

**Metric 5 - Markov Property Verification:**
- Tests if predictions depend only on current state (true Markov property)
- Success criterion: Mean pairwise KL < 0.1

### 3. Results Aggregation: `scripts/aggregate_metrics.py`

Combines all evaluation results into comprehensive CSV files.

```bash
# Aggregate base MM results
python scripts/aggregate_metrics.py \
    --base_dir trainings/MM/MM-100 \
    --output trainings/MM/MM-100/comprehensive_metrics.csv

# Aggregate MM-extra results
python scripts/aggregate_metrics.py \
    --base_dir trainings/MM/MM-100-extra \
    --output trainings/MM/MM-100-extra/comprehensive_metrics.csv
```

### 4. Report Generation: `scripts/generate_sparse_report.py`

Creates markdown reports comparing sparse vs dense models.

```bash
# Generate sparse MM-100 report
python scripts/generate_sparse_report.py --scale 100
```

## Complete Evaluation Workflow

Here's the recommended workflow after training models:

### Step 1: Basic Evaluation

```bash
# For base models (multiple seeds)
python scripts/mm_eval.py 100 --seeds 42 123 999 > trainings/MM/MM-100/eval_summary.txt

# For sparse models
python scripts/mm_eval.py 100-sparse-75 --seeds 42 > trainings/MM/MM-100-sparse-75/eval_summary.txt

# For MM-extra variants
for variant in c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12; do
    python scripts/mm_eval.py $variant > trainings/MM/MM-100-extra/MM-100-extra-$variant-s42/model_eval.txt
done
```

### Step 2: Advanced Metrics (4 & 5)

```bash
# For base models
for seed in 42 123 999; do
    python scripts/mm_eval_transition_matrix.py \
        --transformer_path trainings/MM/MM-100/MM-100-$seed/ckpt.pt \
        --formal_model_path data/mm100/model.pkl \
        --output trainings/MM/MM-100/MM-100-$seed/mm_metrics_4_5.json
done

# For sparse models
python scripts/mm_eval_transition_matrix.py \
    --transformer_path trainings/MM/MM-100-sparse-75/MM-100-sparse-75-42/ckpt.pt \
    --formal_model_path data/mm100-sparse-75/model.pkl \
    --output trainings/MM/MM-100-sparse-75/MM-100-sparse-75-42/mm_metrics_4_5.json
```

### Step 3: Aggregate Results

```bash
# Aggregate all metrics
python scripts/aggregate_metrics.py \
    --base_dir trainings/MM/MM-100 \
    --output trainings/MM/MM-100/comprehensive_metrics.csv

python scripts/aggregate_metrics.py \
    --base_dir trainings/MM/MM-100-sparse-75 \
    --output trainings/MM/MM-100-sparse-75/comprehensive_metrics.csv
```

### Step 4: Generate Reports

```bash
# Generate comparison report
python scripts/generate_sparse_report.py --scale 100
```

## Understanding the Results

### Success Criteria

1. **Perplexity**: Should be within 0.1% of theoretical minimum
2. **Accuracy**: Should be 5-10× random baseline (e.g., 5% for 100 states)
3. **Agreement**: >80% top-1 agreement between different seeds
4. **Transition Fidelity**: KL divergence <0.01 (EXCELLENT) or <0.1 (GOOD)
5. **Markov Property**: Mean pairwise KL <0.1

### Example Results

```
MM-100 Training Results:
- Validation Loss: 4.1899 (Theory: 4.1871)
- Achievement: 99.93% of theoretical optimum
- Perplexity: 66.48
- Accuracy: 5.02% (5× random baseline)
- Transition Fidelity: 0.0089 (EXCELLENT)
- Markov Property: ✓ Perfect (KL < 0.0001)
```

## Using the Legacy Makefile

The Makefile in `extra/` contains targets for running evaluations, but it's deprecated. The manual commands above provide more control and clarity.

## Interpreting Architectural Variations

The MM-extra variants (c1-c12) test different architectural hypotheses:
- **c1-c3**: Varying model sizes
- **c4**: One-hot embeddings
- **c5-c6**: MLP-only architectures
- **c7-c9**: Ablating positional encodings and layer norms
- **c10-c12**: Identity initialization experiments

Key findings:
- MLPs alone can encode transition matrices (c5, c6, c11, c12)
- All architectures learn perfect Markov property
- Sparse models (75% zeros) are learned as effectively as dense models