# nanoMarkov 🎲

A minimal, hackable library for training transformer models on Markov Models (MM). Built for mechanistic interpretability research, based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).


## 🔬 Research Context

This library is part of a larger mechanistic interpretability project exploring how transformers learn formal structures. While LLMs should naturally learn Markov models (a basic sequence structure prevalent in language), we discovered there were no clear recipes or libraries for training and experimenting with this fundamental capability.

nanoMarkov fills this gap by providing a standalone library that addresses the many pitfalls and subtleties in MM training. Originally designed for our research, it's valuable for anyone interested in:

- Understanding transformer internals 
- Testing interpretability tools
- Exploring minimal architectures
- Studying emergence of capabilities

An accompanying paper with full results and analysis is forthcoming.

## 🎯 Goals and Features

- **Training recipes and templates**: Battle-tested configurations for MM training, with proper data preparation and loading
- **Comprehensive evaluation**: 6-metric framework for rigorous MM learning assessment, with detailed discussion of metric significance
- **Easy to modify**: Just a few files, designed to be read, understood and hacked by humans and coding agents
- **Research-focused**: Built for mechanistic interpretability experiments
- **Minimal dependencies**: Based on nanoGPT architecture
- **Throw-away friendly**: Fork it, break it, learn from it!


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/atopoi/nanoMarkov.git
cd nanoMarkov

# Install dependencies
pip install torch numpy matplotlib hmmlearn

# Optional: Install wandb for experiment tracking
pip install wandb
```

### Generate MM Data

```bash
# Generate MM-100 (100 states)
make mm100-data

# Generate MM-10 (small, for testing)
make mm10-data

# Generate sparse MM (75% zero transitions)
make mm100-sparse-75-data
```

### Train a Model

```bash
# Train on MM-100
make mm100-train

# Train minimal architecture (fast)
make mm100-train TRAIN=train_mm.py

# Train with different seed
make mm100-train-123

# Train with WandB logging enabled
make mm100-train WANDB=True
```

### Experiment Tracking with WandB

WandB (Weights & Biases) integration is **disabled by default** to keep the library minimal. To enable:

```python
# In your config file or command line
wandb_log = True
wandb_project = 'nanomarkov'
wandb_run_name = 'mm100-experiment'
```

Or via command line:
```bash
python train_mm.py config/mm100.py --wandb_log=True
```

### Evaluate

```bash
# Run comprehensive evaluation
make mm100-eval

# Evaluate specific checkpoint
python scripts/mm_eval.py --ckpt_path trainings/MM/MM-100/MM-100-42/ckpt.pt
```

## 📊 Key Features

### Markov Model Generation
- Uses `hmmlearn` library for mathematically correct MM generation
- Supports dense and sparse transition matrices
- Configurable state spaces (10 to 1000+ states)
- Dirichlet-sampled transitions for diverse dynamics

### Training Pipeline
- **Minimal implementation**: ~700 lines of training code. Based on nanoGPT's. Easy to read by humans and LLMs.
- **Fast iteration**: Train MM-100 in ~30 minutes on consumer hardware
- **Flexible configs**: Easy to modify architectures and hyperparameters
- **Optional WandB**: Experiment tracking disabled by default, easy to enable

### 6-Metric Evaluation Framework

1. **Cross-Entropy Loss**: Direct optimization objective
2. **Perplexity**: Model uncertainty measure
3. **Accuracy**: Next-token prediction accuracy
4. **Agreement**: Inter-seed consistency check
5. **Transition Matrix Fidelity**: KL divergence from ground truth
6. **Markov Property Verification**: Tests true memorylessness

### Key Results
- **Near-perfect learning**: 99.93% of theoretical optimum on MM-100
- **MLPs are enough**: MLP-only version can encode the transitions matrices
- **Perfect Markov compliance**: All architectures learn true Markov property


## 📁 Project Structure

```
nanoMarkov/
├── train_mm.py                       # Minimal training script (~700 lines)
├── calculate_theoretical_minimums.py # Helper for theoretical baselines
├── configurator.py                   # Config file runner
├── model.py                          # GPT model (from nanoGPT)
├── scripts/                          # Data generation & evaluation scripts
│   ├── mm_generate.py
│   ├── mm_eval.py
│   └── …                             # Other evaluation utilities
├── configs/                          # Configuration generators
│   ├── mm_config.py
│   └── mm_config_extra.py
├── Makefile                          # Convenient commands
├── data/                             # Generated MM datasets
└── trainings/                        # Training outputs and checkpoints
```

## 🛠️ Hacking Guide

### Modify Architecture
Edit configs or create new ones:
```python
# configs/mm100_tiny.py
n_layer = 1
n_head = 1
n_embd = 64
```

### Add New Metrics
Extend `scripts/mm_eval.py`:
```python
def compute_my_metric(model, mm_model, data):
    # Your metric computation
    return result
```

### Change Training Dynamics
Modify `train_mm.py`:
- Adjust learning rate schedules
- Add regularization
- Implement new loss functions

## 📈 Example Results

```
MM-100 Training Results:
- Validation Loss: 4.1899 (Theory: 4.1871)
- Achievement: 99.93% of theoretical optimum
- Perplexity: 66.48
- Accuracy: 5.02% (5× random baseline)
- Markov Property: ✓ Perfect (KL < 0.0001)
```

## 🤝 Contributing

This is research code - feel free to:
- Fork and modify for your experiments
- Report interesting findings
- Suggest minimal improvements
- Share your interpretability insights

## 📚 Citation

If you use nanoMarkov in your research, please cite:
```bibtex
@software{nanomarkov2025,
  author = {Zolfaghari, Houman},
  title = {nanoMarkov: Minimal Markov Model Training for Transformers},
  year = {2025},
  url = {https://github.com/atopoi/nanoMarkov}
}
```

## 🙏 Acknowledgments

- Andrej Karpathy for [nanoGPT](https://github.com/karpathy/nanoGPT)
- The `hmmlearn` developers for robust MM implementation
- The mechanistic interpretability community

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

This work derives from Andrej Karpathy’s nanoGPT. We’ve included the original MIT license text
for nanoGPT in `third_party/nanoGPT/LICENSE`.

To comply with nanoGPT’s license, please preserve its copyright notice and
license when reusing or modifying any nanoGPT-derived code.

---

*Built with ❤️ for interpretability research. Keep it simple, keep it hackable!*
