"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
from collections import deque
import pickle
from contextlib import nullcontext

import numpy as np
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
# STREAMLINED: No external evaluation imports needed

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
resume_tokens = 0 # number of tokens seen before resume (in thousands) - used when checkpoint doesn't have token_count
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
data_dir = None  # if None, defaults to 'data/{dataset}', otherwise use custom path
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# cyclical batch size schedule
batch_size_cycle_enabled = False  # enable cyclical batch size schedule
batch_size_cycle_length = 10000   # cycle length in iterations
batch_size_cycle_reduced_fraction = 0.1  # fraction of cycle with reduced batch size (10% = 1000 iters)
batch_size_cycle_reduced = 2      # batch size during reduced phase (regular phase uses normal batch_size)
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# model metadata
model_name = 'tinystories-small'  # name of the model
model_arch = 'gpt'  # architecture type
model_shape = f'{n_layer}L{n_head}H_{n_embd}D'  # model shape in format layers_heads_dimension
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 80000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
lr_decay_linear=False # HZ experimenting with linear decay
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system - auto-detection defaults (can be overridden in config)
device = 'auto' # 'auto' (recommended), 'cuda', 'mps', 'cpu', or specific like 'cuda:0'
dtype = 'auto'  # 'auto' (recommended), 'float32', 'bfloat16', 'float16'
compile = False # seems best even with cuda, on mps complie creates problems. 'auto', True, False
train_seed = 1337 # training random seed (can be overridden in config)
# added
lm_head_project = False # using a projection matrix before the last unembedding 
lm_head_tied = True
mlp_key_bias = False  # bias in the first layer or the MLPs
mlp_filter   = 'GELU' # GELU, other choices: RELU, RELU2 (relu squared), RELU3 (relu exp 3)
mlp_ratio    = 4      # MLP hidden dimension ratio (hidden_dim = mlp_ratio * n_embd)
reluify      = False # use reluification - https://arxiv.org/pdf/2310.04564.pdf
use_ln       = True
mlp_only     = False
mlp_first_matrix_identity = False
# interpretability: Top-K Prediction View tool defaults
topk_view    = False           # enable top-k prediction analysis on eval batches
topk_k       = 10              # number of top predictions to record
topk_layers  = 'final'         # layers to inspect (comma-separated indices or 'final')
topk_output  = 'interpret/topk.jsonl'  # relative path under out_dir for JSONL output
# are we using absolute positional embeddings? TODO replace by a choice of positional embeddings or none
pos_ape      = True
# use frozen one-hot embeddings (requires vocab_size <= n_embd)
one_hot_embed = False

# MM-specific: No completion generation needed
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # Log input and output file paths
    print(f"Train data file: data/{dataset}/train.bin", file=sys.stderr)
    print(f"Validation data file: data/{dataset}/val.bin", file=sys.stderr)
torch.manual_seed(train_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# Auto device, dtype, and compilation detection
print("=== DEVICE & PERFORMANCE CONFIGURATION ===")

# Device detection
if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
        device_type = 'cuda'
        print(f"✓ Device: CUDA GPU detected and selected")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        device_type = 'mps'
        print(f"✓ Device: MPS (Apple Silicon) detected and selected")
    else:
        device = 'cpu'
        device_type = 'cpu'
        print(f"✓ Device: CPU fallback (no GPU detected)")
else:
    device_type = 'mps' if device == 'mps' else 'cuda' if 'cuda' in device else 'cpu'
    print(f"✓ Device: {device} (manually specified)")

# Dtype detection based on device capabilities
if dtype == 'auto':
    if device_type == 'cuda':
        if torch.cuda.is_bf16_supported():
            dtype = 'bfloat16'
            print(f"✓ Dtype: bfloat16 (CUDA supports BF16)")
        else:
            dtype = 'float16'
            print(f"✓ Dtype: float16 (CUDA fallback)")
    elif device_type == 'mps':
        dtype = 'float32'
        print(f"✓ Dtype: float32 (MPS stability requirement)")
    else:  # cpu
        dtype = 'float32'
        print(f"✓ Dtype: float32 (CPU default)")
else:
    print(f"✓ Dtype: {dtype} (manually specified)")

# Compilation detection based on device and stability
if compile == 'auto':
    if device_type == 'cuda':
        compile = False
        print(f"✓ Compile: enabled (CUDA optimization)")
    elif device_type == 'mps':
        compile = False
        print(f"✓ Compile: disabled (MPS compatibility)")
    else:  # cpu
        compile = False
        print(f"✓ Compile: disabled (CPU compatibility)")
else:
    print(f"✓ Compile: {compile} (manually specified)")

print(f"FINAL CONFIG: device={device}, dtype={dtype}, compile={compile}")
print("=" * 45)
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# count tokens for model comparison
token_count = 0

# MM-specific: Removed chess variables

# cyclical batch size scheduler
def get_batch_size_for_iter(iter_num, cycle_length, reduced_fraction, batch_size_normal, batch_size_reduced):
    """
    Returns the batch size for the given iteration based on cyclical schedule.
    
    Args:
        iter_num: Current iteration number
        cycle_length: Length of each cycle (e.g., 10000)
        reduced_fraction: Fraction of cycle with reduced batch size (e.g., 0.1 for 10%)
        batch_size_normal: Normal batch size (e.g., 20)
        batch_size_reduced: Reduced batch size (e.g., 2)
    
    Returns:
        int: Batch size to use for this iteration
    """
    cycle_position = iter_num % cycle_length
    reduced_phase_length = int(cycle_length * reduced_fraction)
    
    if cycle_position < reduced_phase_length:
        return batch_size_reduced
    else:
        return batch_size_normal

# poor man's data loader
if data_dir is None:
    data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split, batch_size_override=None):
    """
    Get a batch of data for training or validation.
    
    Args:
        split: 'train' or 'val'
        batch_size_override: Override the global batch_size for this batch (used for cyclical scheduling)
    """
    data = train_data if split == 'train' else val_data
    effective_batch_size = batch_size_override if batch_size_override is not None else batch_size
    ix = torch.randint(len(data) - block_size, (effective_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    global token_count
    token_count += (block_size * effective_batch_size) // 1000
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

config_keys = [
    'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
    'dropout',
    'use_ln',
    'lm_head_tied',
    'lm_head_project',
    'mlp_only',
    'mlp_key_bias',
    'mlp_filter',
    'mlp_ratio',
    'mlp_first_matrix_identity',
    'reluify',
    'pos_ape',
    'one_hot_embed']

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None,
                  dropout=dropout,
                  use_ln = use_ln,
                  lm_head_tied = lm_head_tied, 
                  lm_head_project = lm_head_project,
                  mlp_only = mlp_only,
                  mlp_key_bias = mlp_key_bias,
                  mlp_filter = mlp_filter,
                  mlp_ratio = mlp_ratio,
                  mlp_first_matrix_identity = mlp_first_matrix_identity,
                  reluify = reluify,
                  pos_ape= pos_ape,
                  one_hot_embed = one_hot_embed) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in config_keys:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
        else:
            print(f"arg not found in saved checkpoint: {k} using {config[k]}")
            model_args[k] = config[k]
    print(model_args)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        if k == 'transformer.lm_head.weight':
            print("Found lm_head")
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter']
    best_val_loss = checkpoint['best_val_loss']
    # Load token_count from checkpoint, fall back to resume_tokens if not present
    token_count = checkpoint.get('tk', resume_tokens)
    if token_count == 0 and resume_tokens > 0:
        token_count = resume_tokens
        print(f"Using --resume_tokens value: {token_count}k tokens")
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in config_keys:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# MM-specific: Removed chess detection functions and initialization

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    if lr_decay_linear:
        return  learning_rate - decay_ratio * (learning_rate - min_lr)
    else:
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
# determine initial batch size
if batch_size_cycle_enabled:
    initial_batch_size = get_batch_size_for_iter(
        0, batch_size_cycle_length, batch_size_cycle_reduced_fraction,
        batch_size, batch_size_cycle_reduced
    )
else:
    initial_batch_size = batch_size
X, Y = get_batch('train', initial_batch_size) # fetch the very first batch

# vocabulary usage tracking
vocab_usage = np.zeros(model.config.vocab_size, dtype=np.int64)
vocab_stats_file = os.path.join(out_dir, 'vocab_usage.npy')


""" 
print(f"gragradient_accumulation_steps: {gradient_accumulation_steps}")

print(f"X,Y: {X.shape} {Y.shape}")
print(X)

import tiktoken
enc = tiktoken.get_encoding("gpt2")
#enc.encode(s, allowed_special={"<|endoftext|>"})
print(f"Model proj: {model.transformer.unembed_proj.weight.shape}")
print(f"Model h0: {model.transformer.h[0].attn.c_attn.weight.shape}")
print(f"Model pos emb: {model.transformer.wpe.weight.shape}")
for i in range(20) :
    print(f"--batch{i}: {X[i]}")
    print(X[i])
    print(X[i].shape)
    print(enc.decode(X[i].tolist())) """

t0 = time.time()
last_losses=deque([0,0,0,0,0])
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# MM-specific: Removed completion generation function (lines 557-975)
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # determine current batch size based on cyclical schedule
    if batch_size_cycle_enabled:
        current_batch_size = get_batch_size_for_iter(
            iter_num, 
            batch_size_cycle_length, 
            batch_size_cycle_reduced_fraction,
            batch_size, 
            batch_size_cycle_reduced
        )
        # log batch size changes
        cycle_position = iter_num % batch_size_cycle_length
        if cycle_position == 0 and iter_num > 0 and master_process:
            print(f"iter {iter_num}: Cycle {iter_num // batch_size_cycle_length} - Starting reduced batch size phase (batch_size={current_batch_size})")
        elif cycle_position == int(batch_size_cycle_length * batch_size_cycle_reduced_fraction) and master_process:
            print(f"iter {iter_num}: Switching to regular batch size phase (batch_size={current_batch_size})")
    else:
        current_batch_size = batch_size
    
    # freeze the model weights progressively
    #if iter_num == 4000:
    #    print(f"Freezing the unembedding projection matrix at step{iter_num}")
    #    model.transformer.unembed_proj.weight.requires_grad = False
    #if iter_num == 10000:
    #    print(f"Freezing the layer 0 qkv matrix at step{iter_num}")
    #    model.transformer.h[0].attn.c_attn.weight.requires_grad = False

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # MM-specific: No completion generation needed
        if wandb_log:
            # calculate vocabulary statistics
            unique_tokens_used = np.count_nonzero(vocab_usage)
            total_tokens_seen = np.sum(vocab_usage)
            vocab_coverage = unique_tokens_used / model.config.vocab_size
            
            # Base logging data
            log_data = {
                "iter": iter_num,
                'tk': token_count,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "vocab/unique_tokens": unique_tokens_used,
                "vocab/total_tokens": total_tokens_seen,
                "vocab/coverage": vocab_coverage,
            }
            
            # MM-specific: Removed chess statistics
            # Add distance metrics if available (only during early training)
            if iter_num < 10000:
                up_dist, l0_dist, um_dist = model.get_component_change_rate()
                log_data.update({
                    "up_dist": up_dist,
                    "l0_dist": l0_dist,
                    "um_dist": um_dist
                })
            
            # MM-specific: Removed chess move perplexity
            wandb.log(log_data)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter': iter_num,
                    'tk': token_count,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                
                # Verify one-hot embeddings before saving (if applicable)
                if hasattr(raw_model.config, 'one_hot_embed') and raw_model.config.one_hot_embed:
                    try:
                        raw_model._verify_one_hot_embeddings()
                    except ValueError as e:
                        print(f"❌ One-hot embedding verification failed: {e}")
                        print(f"🚨 CRITICAL: Model integrity compromised, checkpoint may be invalid!")
                        # Continue saving but warn user
                
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                # save vocabulary usage stats alongside checkpoint
                np.save(vocab_stats_file, vocab_usage)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # track vocabulary usage from current batch
        if X is not None:
            tokens = X.cpu().numpy().flatten()
            for token in tokens:
                vocab_usage[token] += 1
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', current_batch_size)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        last_losses.append(lossf)
        last_losses.popleft()
        rolling_loss = 0
        for v in last_losses: rolling_loss = rolling_loss + v
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(current_batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        batch_info = f", bs {current_batch_size}" if batch_size_cycle_enabled else ""
        
        # MM-specific: Removed chess moves info
        print(f"iter {iter_num}: loss {lossf:.4f}, avg5 {rolling_loss/5:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr: {lr:.6f}, tk {token_count}{batch_info}")
        if iter_num % 20 ==0 and iter_num < 10000 and iter_num % eval_interval != 0:
            up_dist, l0_dist, um_dist = model.get_component_change_rate()
            print(f"dist: {iter_num}: up_dist {up_dist:.4e}, l0_dist {l0_dist:.4e}, um_dist {um_dist:.4e}")
            if wandb_log:
                dist_log_data = {
                    "iter": iter_num,
                    'tk': token_count,
                    "up_dist": up_dist,
                    "l0_dist": l0_dist,
                    "um_dist": um_dist
                }
                
                # MM-specific: Removed chess moves tracking
                wandb.log(dist_log_data)
        sys.stdout.flush()
        if wandb_log:
            st_log_data = {
                "iter": iter_num,
                'tk': token_count,
                "stloss": lossf,
                "stlr": lr
            }
            
            wandb.log(st_log_data)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Clean up DDP group if used
if ddp:
    destroy_process_group()
# After training completes, list key outputs for the master process
if master_process:
    print("\nSmoke Test Complete. Output files in run directory:")
    for fname in sorted(os.listdir(out_dir)):
        print(f" - {fname}")
