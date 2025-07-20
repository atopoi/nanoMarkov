"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import json
import tiktoken

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        use_relu = config.reluify or config.mlp_filter == "RELU" or config.mlp_filter == "RELU2" or config.mlp_filter == "RELU3"
        mlp_hidden_dim = config.mlp_ratio * config.n_embd
        #self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias= config.mlp_key_bias or config.bias)
        if config.mlp_first_matrix_identity:
            self.c_fc = None  # Skip first matrix for identity operation
        else:
            self.c_fc = nn.Linear(config.n_embd, mlp_hidden_dim, bias=False)
        self.filter  = nn.ReLU() if use_relu else nn.GELU()
        self.c_proj  = nn.Linear(mlp_hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.shift1  = config.mlp_key_bias
        self.mlp_filter = config.mlp_filter
        #if config.mlp_key_bias:
        #    self.c_fc.bias.data.fill_(-0.01)
        # Initialization log suppressed for cleaner output

    #def forward(self, x):
    #    x = self.c_fc(x)
    #    x = self.filter(x)
    #    x = self.c_proj(x)
    #    x = self.dropout(x)
    #    return x
    
    def forward(self, x):
        if self.c_fc is not None:
            x = self.c_fc(x)
        if self.shift1:
            x -= 0.002 # fixed relu shift
        x = self.filter(x)
        if self.mlp_filter == "RELU2":
            x = x ** 2
        if self.mlp_filter == "RELU3":
            x = x ** 3
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_ln   = config.use_ln
        self.mlp_only = config.mlp_only
        self.reluify  = config.reluify
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = None if self.mlp_only else CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)
        self.relu = nn.ReLU()

    def forward(self, x):
        xn = x
        if not self.mlp_only:
            if self.use_ln:
                xn = self.ln_1(x)
            if self.reluify:
                xn = self.relu(xn)
            x = x + self.attn(xn)
        if self.use_ln:
            xn = self.ln_2(x)
        if self.reluify:
            xn = self.relu(xn)
        x = x + self.mlp(xn)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768
    dropout:    float = 0.0
    bias:          bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    lm_head_tied:  bool = True,   # True tied the lm_head (unembedding matrix) to the vocabulary
    lm_head_project: bool = False # Use a projection matrix before unembedding
    mlp_key_bias:  bool = False   # True : bias only in the first kayer of the mlps
    reluify:       bool = False   # experiments with reluification https://arxiv.org/pdf/2310.04564.pdf
    use_ln:        bool = True
    mlp_only:      bool = False
    mlp_filter:    str  = 'GELU'  # Declaring mlp_filter as a string
    mlp_ratio:     int  = 4       # MLP hidden dimension ratio (hidden_dim = mlp_ratio * n_embd)
    mlp_first_matrix_identity: bool = False  # Use identity for first MLP matrix (skip first linear layer)
    pos_ape:       bool = True    # Are we using absolute positional embeddings? TODO replace by a choice of positional embeddings or none
    one_hot_embed: bool = False   # Use frozen one-hot embeddings (requires vocab_size <= n_embd)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Validate one-hot embedding constraints
        if config.one_hot_embed:
            assert config.vocab_size <= config.n_embd, f"one_hot_embed requires vocab_size ({config.vocab_size}) <= n_embd ({config.n_embd})"
            print(f"🔥 Using one-hot frozen embeddings: vocab_size={config.vocab_size}, n_embd={config.n_embd}")

        self.prev_unembed_proj = None # for change tracking
        self.prev_l0 = None    # for change tracking
        self.relu = nn.ReLU()

        print(config)

        # Choose embedding type
        if config.one_hot_embed:
            # Create one-hot embedding layer
            wte = nn.Embedding(config.vocab_size, config.n_embd)
            # Initialize with one-hot vectors and freeze
            with torch.no_grad():
                wte.weight.zero_()
                for i in range(config.vocab_size):
                    wte.weight[i, i] = 1.0
            wte.weight.requires_grad = False  # Freeze the embedding
        else:
            wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            wpe = nn.Embedding(config.block_size, config.n_embd) if config.pos_ape else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) if not config.lm_head_tied else None,
            unembed_proj = nn.Linear(config.n_embd, config.n_embd, bias=False) if config.lm_head_project else None
        ))
        if config.lm_head_tied:
            ## tied version: lm_head is tied to the vocabulary embeddings so it's not saved 
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #else:
        #    self.lm_head = self.transformer.lm_head

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if config.lm_head_tied:
            if config.one_hot_embed:
                # For one-hot embeddings, we need to be careful about weight tying
                # The embedding is already frozen, so we should copy its values to lm_head
                # but NOT share the tensor (to avoid initialization corruption)
                with torch.no_grad():
                    self.lm_head.weight.copy_(self.transformer.wte.weight.T)
                self.lm_head.weight.requires_grad = False  # Also freeze the LM head
            else:
                self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Re-verify one-hot embeddings after all initialization (they might have been corrupted by LM head tying)
        if hasattr(config, 'one_hot_embed') and config.one_hot_embed:
            self._verify_one_hot_embeddings()

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    # TDO hz improve :)
    @torch.no_grad()
    def get_component_change_rate(self):
        D = self.config.n_embd
        V = self.config.vocab_size
        up_dist = 0
        l0_dist = 0
        um_dist = 0

        # if lm_head_project calculate the project matrix change
        if self.config.lm_head_project:
            unembed_proj = self.transformer.unembed_proj.weight.data.cpu().numpy()
            print(unembed_proj[0:3,0:3])
            _, S, _ = np.linalg.svd(unembed_proj)
            print(S[0:10])
            if self.prev_unembed_proj is not None:
                up_dist = np.linalg.norm(self.prev_unembed_proj - unembed_proj) / math.sqrt(D*D)
            else:
                up_dist = np.linalg.norm(unembed_proj) / D*D
            self.prev_unembed_proj = unembed_proj

        # first layer all head matrices
        if not self.config.mlp_only:
            l0 = self.transformer.h[0].attn.c_attn.weight.data.cpu().numpy()
            if self.prev_l0 is not None:
                l0_dist = np.linalg.norm(self.prev_l0 - l0) / math.sqrt(D*D*3)
            else:
                l0_dist = np.linalg.norm(l0) / D*D*3
            self.prev_l0 = l0

        if not self.config.lm_head_tied:
            um_dist = np.linalg.norm(self.transformer.wte.weight.data.cpu().numpy() - self.transformer.lm_head.weight.data.cpu().numpy()) / D*V

        return up_dist, l0_dist, um_dist

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _verify_one_hot_embeddings(self):
        """Verify that one-hot embeddings are still identity matrix and frozen"""
        if not (hasattr(self.config, 'one_hot_embed') and self.config.one_hot_embed):
            return
        
        embeddings = self.transformer.wte.weight
        vocab_size = self.config.vocab_size
        
        # Check if frozen
        if embeddings.requires_grad:
            raise ValueError(f"One-hot embeddings should be frozen (requires_grad=False), but requires_grad={embeddings.requires_grad}")
        
        # Check if identity matrix
        expected_identity = torch.eye(vocab_size, device=embeddings.device, dtype=embeddings.dtype)
        if embeddings.shape != (vocab_size, vocab_size):
            raise ValueError(f"One-hot embeddings should be square ({vocab_size}, {vocab_size}), but got {embeddings.shape}")
        
        frobenius_diff = torch.norm(embeddings - expected_identity, p='fro').item()
        if frobenius_diff > 1e-6:
            raise ValueError(f"One-hot embeddings deviated from identity matrix (Frobenius norm: {frobenius_diff:.8f})")
        
        # Verify diagonal and off-diagonal values
        diagonal = torch.diag(embeddings)
        off_diagonal_mask = ~torch.eye(vocab_size, dtype=bool, device=embeddings.device)
        off_diagonal_values = embeddings[off_diagonal_mask]
        
        if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6):
            raise ValueError(f"One-hot embedding diagonals should be 1.0, got mean={diagonal.mean():.8f}, std={diagonal.std():.8f}")
        
        if not torch.allclose(off_diagonal_values, torch.zeros_like(off_diagonal_values), atol=1e-6):
            raise ValueError(f"One-hot embedding off-diagonals should be 0.0, got mean={off_diagonal_values.mean():.8f}, std={off_diagonal_values.std():.8f}")
        
        print(f"✅ One-hot embeddings verified: Identity matrix ({vocab_size}x{vocab_size}), frozen, Frobenius diff: {frobenius_diff:.2e}")
        
        # Also verify lm_head is identity when tied
        if hasattr(self.config, 'lm_head_tied') and self.config.lm_head_tied and hasattr(self, 'lm_head'):
            self._verify_lm_head_identity()

    def _verify_lm_head_identity(self):
        """Verify that lm_head is identity matrix when tied to one-hot embeddings"""
        vocab_size = self.config.vocab_size
        lm_head_weight = self.lm_head.weight
        
        # Check shape
        if lm_head_weight.shape != (vocab_size, vocab_size):
            raise ValueError(f"LM_head shape should be ({vocab_size}, {vocab_size}) for tied one-hot embeddings, got {lm_head_weight.shape}")
        
        # Create identity matrix
        identity_matrix = torch.eye(vocab_size, device=lm_head_weight.device, dtype=lm_head_weight.dtype)
        
        # Check if it's identity
        frobenius_diff = torch.norm(lm_head_weight - identity_matrix, p='fro').item()
        
        if not torch.allclose(lm_head_weight, identity_matrix, atol=1e-6):
            print(f"❌ LM_head verification FAILED!")
            print(f"   Expected: Identity matrix")
            print(f"   Got: Frobenius difference = {frobenius_diff:.6f}")
            print(f"   Max difference: {(lm_head_weight - identity_matrix).abs().max():.6f}")
            print(f"   LM_head requires_grad: {lm_head_weight.requires_grad}")
            print(f"   Sample values: {lm_head_weight[0, :5]}")
            raise ValueError(f"LM_head should be identity matrix when tied to one-hot embeddings, Frobenius diff: {frobenius_diff:.6f}")
        
        # Check frozen status
        if lm_head_weight.requires_grad:
            raise ValueError(f"LM_head should be frozen (requires_grad=False) when tied to one-hot embeddings, got requires_grad={lm_head_weight.requires_grad}")
        
        print(f"✅ LM_head verified: Identity matrix ({vocab_size}x{vocab_size}), frozen, Frobenius diff: {frobenius_diff:.2e}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Skip initialization for tied lm_head when using one-hot embeddings
            if (hasattr(self.config, 'lm_head_tied') and self.config.lm_head_tied and 
                hasattr(self.config, 'one_hot_embed') and self.config.one_hot_embed and 
                hasattr(self, 'lm_head') and module is self.lm_head):
                # lm_head is tied to one-hot embeddings - already initialized, skip
                print("🔧 Skipping lm_head initialization - tied to one-hot embeddings")
                return
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Skip initialization for one-hot embeddings (already set up correctly)
            if hasattr(self.config, 'one_hot_embed') and self.config.one_hot_embed and module is self.transformer.wte:
                # Verify one-hot embeddings are still correct and frozen
                self._verify_one_hot_embeddings()
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_tokens(self, idx):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.pos_ape:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            tok_emb = tok_emb + pos_emb
        return tok_emb

    def forward(self, idx, targets=None, analyze_prompt=False):
        # forward the GPT model itself
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        tok_emb = self.embed_tokens(idx)
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        
        if self.config.use_ln:
            x = self.transformer.ln_f(x)
        if self.config.reluify:
            x = self.relu(x)
        
        # inference-time mini-optimization: only forward the lm_head on the very last position
        # unless we're analyzing the prompt, in which case we need all positions
        if targets is None and not analyze_prompt:
            #print(x.shape)
            x = x[:, [-1], :]
        
        # with a projection matrix before unembedding
        if self.config.lm_head_project:
            x = self.transformer.unembed_proj(x)
        
        # match against the vocabulary embeddings
        if self.config.lm_head_tied:
            ##  tied version, lm_head is simply the tokens embeddings matrix
            logits = self.lm_head(x)
        else:
            ## untied version
            logits = self.transformer.lm_head(x)

        loss = None
        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.pos_ape:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    @torch.no_grad()
    def topk_predictions(self, idx, next_ids=None, topk_k=10, topk_layers=('final',)):
        """
        Return top-K next-token predictions at specified layers. Does not modify model state.
        Args:
            idx: LongTensor[B, T], input token IDs.
            next_ids: LongTensor[B, T], true next token IDs (optional).
            topk_k: int, number of top predictions to return.
            topk_layers: list of layer indices (int) or 'final'.
        Returns:
            List of records (dict) with keys: layer, position, input_tokens, true_next, true_rank, true_prob, top_predictions.
        """
        device = idx.device
        B, T = idx.size()
        # Embed tokens and positions
        hs = self.embed_tokens(idx)
        hs = self.transformer.drop(hs)
        # Collect hidden states
        hidden_states = {}
        for i, block in enumerate(self.transformer.h):
            hs = block(hs)
            if i in topk_layers:
                hidden_states[i] = hs.clone()
        # Final layer
        if 'final' in topk_layers:
            h_final = hs
            if self.config.use_ln:
                h_final = self.transformer.ln_f(h_final)
            if self.config.reluify:
                h_final = self.relu(h_final)
            hidden_states['final'] = h_final.clone()
        # Get unembed weight matrix
        weight = self.lm_head.weight if hasattr(self, 'lm_head') else self.transformer.lm_head.weight
        # Token decoder
        enc = tiktoken.get_encoding('gpt2')
        records = []
        # Iterate over layers and positions
        for layer, h in hidden_states.items():
            # Project to logits
            logits = F.linear(h, weight)
            probs = F.softmax(logits, dim=-1)
            for b in range(B):
                # Decode input tokens once per example
                input_tokens = [enc.decode([int(tok)]) for tok in idx[b].tolist()]
                for t in range(T):
                    rec = {'layer': layer, 'position': t, 'input_tokens': input_tokens}
                    if next_ids is not None:
                        true_id = int(next_ids[b, t])
                        prob_row = probs[b, t]
                        logit_row = logits[b, t]
                        rec['true_next'] = enc.decode([true_id])
                        rec['true_prob'] = float(prob_row[true_id])
                        rec['true_rank'] = int((logit_row > logit_row[true_id]).sum().item()) + 1
                    # Top-K predictions
                    topk_probs, topk_inds = probs[b, t].topk(topk_k)
                    rec['top_predictions'] = [
                        {'token': enc.decode([int(tok)]), 'prob': float(prob)}
                        for prob, tok in zip(topk_probs.tolist(), topk_inds.tolist())
                    ]
                    records.append(rec)
        return records

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, verbose=False, analyze_prompt=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        details = { "probs": [], "choices": [], "idx_notopk": [], "next_ids": [] }
        
        # Analyze prompt if requested
        if analyze_prompt:
            # Get the full logits for the entire prompt sequence
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits_full, _ = self(idx_cond, analyze_prompt=True)
            
            # For each position in the prompt, get the best next token and its probability
            prompt_next = []
            prompt_next_prob = []
            
            for pos in range(logits_full.size(1)):
                # Get logits for this position (first batch item only)
                logits_pos = logits_full[0, pos, :]  # shape: (vocab_size,)
                # Apply softmax to get probabilities
                probs_pos = F.softmax(logits_pos, dim=-1)  # shape: (vocab_size,)
                # Get the best token and its probability
                best_prob, best_token = torch.max(probs_pos, dim=-1)
                prompt_next.append(best_token.item())
                prompt_next_prob.append(best_prob.item())
            
            details["prompt_next"] = prompt_next
            details["prompt_next_prob"] = prompt_next_prob
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert all logits to (normalized) probabilities
            probs_all = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs_all, num_samples=1)

            # optionally crop the logits to only the top k options
            if verbose or (top_k is not None):
                k = 10 if top_k is None else top_k
                k = min(k, logits.size(-1))
                best_logits, best_ids = torch.topk(logits, k)
                # apply softmax to convert top k logits to (normalized) probabilities
                probs_topk = F.softmax(best_logits, dim=-1)
                # sample
                selected_topk = torch.multinomial(probs_topk, num_samples=1)  # index among top k
                id_next_topk  = best_ids[0,selected_topk] # the id

                # old version for the non-verbose case but there is a strange bug when running on mac
                #if top_k is not None:
                #    logits[logits < best_logits[:, [-1]]] = -float('Inf')
                prob_best_10 = probs_all[best_ids[0]]
                prob_next_topk = probs_topk[0,selected_topk]
                details["choices"].append(list(zip(best_ids[0].tolist(), best_logits[0].tolist(), prob_best_10.tolist(), prob_next_topk.tolist())))
            
            # append sampled index to the running sequence and continue
            if top_k is not None:
                details["idx_notopk"].append(idx_next.item())
                details["probs"].append(prob_next_topk.item())
                idx_next = id_next_topk
            else:
                # record probability of selected token (batch size 1 assumed)
                # idx_next is shape (batch,1); extract scalar index
                token_idx = idx_next.item() if idx_next.numel() == 1 else idx_next[0, -1].item()
                # probs_all shape: (batch, vocab_size)
                prob = probs_all[0, token_idx].item()
                details["probs"].append(prob)
                details["next_ids"].append(token_idx)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, details
    
    ## adapted from  https://github.com/JiajunSong629/uncover-hidden-geometry see https://arxiv.org/pdf/2310.04861.pdf
    @torch.no_grad()
    def generate_hiddens(self, idx):
        """
        For a batch of indices idx, generate the hidden states at each layer for further
        analysis. Mostly you'll want to make sure to be in model.eval() mode of operation for this.
        Plus the embedding layer there are 1 + n_layer hidden states, with each of size
        (batch_size, seq_len, d_model). We will stack the layer index at 0th dimension, and this gives
        a (1 + n_layer, batch_size, seq_len, d_model) torch.tensor.
        """
        device = idx.device
        b, t = idx.size()
        self.to(device)

        # forward the GPT model itself
        #tok_emb = self.embed_tokens(idx)
        #tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) ??? #CHECK HZ
        #pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        #x = tok_emb + pos_emb
        tok_emb = self.embed_tokens(idx)

        x = self.transformer.drop(x)
        hiddens = [x.numpy(force=True)]
        for block in self.transformer.h:
            x = block(x)
            hiddens.append(x.numpy(force=True))

        # not necessary?
        #if self.config.use_ln:
        #    x = self.transformer.ln_f(x)
        #if self.config.reluify:
        #    x = self.relu(x)

        hiddens = np.array(hiddens)
        return hiddens
