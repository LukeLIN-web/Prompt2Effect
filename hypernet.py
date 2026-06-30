"""Prompt2Effect HyperNet.

A Perceiver-style hypernetwork that maps a text embedding and a base weight matrix
to a low-rank (LoRA) update. The base weight is encoded as row/column tokens,
compressed into a small set of latent tokens that attend to the text condition,
and then decompressed to predict the LoRA factors ``A`` and ``B``.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class HyperConfig:
    text_dim: int = 4096
    weight_dim: int = 4096  # dimension of the original weight
    # Maps module_name -> spec; listed after extracting the model's weights.
    module_specs: Optional[Dict[str, str]] = None

    # Transformer dims
    hidden_dim: int = 1024  # Transformer d_model
    n_heads: int = 8
    n_layer: int = 8
    dropout: float = 0.1

    # LoRA
    rank: int = 128
    alpha_scale: float = 1.0  # multiply after Softplus

    # Number of latent tokens in the Perceiver-style compressor/decompressor
    num_latents: int = 256

    # Optional conditioning on (layer_idx, module_type)
    include_emb: bool = True
    max_layers: int = 32
    layer_embed_dim: int = 32
    module_embed_dim: int = 32
    use_sin_pos: bool = True
    grad_ckpt: bool = False


def sinusoidal_positions(L: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard sinusoidal positional encodings of shape ``[L, d]``."""
    if L <= 0:
        return torch.zeros(0, d, device=device, dtype=torch.float32).to(dtype)
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    i = torch.arange(d, device=device, dtype=torch.float32)
    # Frequencies on even dims; odd dims share the same frequency.
    div_term = torch.exp(-(math.log(10000.0) * (i // 2) * 2 / d))
    angles = pos * div_term
    pe = torch.zeros(L, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.to(dtype)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SelfAttnBlock(nn.Module):
    """Pre-Norm self-attention block: x -> LN -> MHA(x, x, x) -> + -> LN -> FF -> +."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: [B, S], True = PAD/ignore (PyTorch convention).
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class CrossAttnBlock(nn.Module):
    """Pre-Norm cross-attention block: q -> LN -> MHA(q, ctx, ctx) -> + -> LN -> FF -> +."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(
        self,
        q: torch.Tensor,                                     # [B, Q, d]
        ctx: torch.Tensor,                                   # [B, K, d]
        ctx_key_padding_mask: Optional[torch.Tensor] = None,  # [B, K], True = PAD/ignore
    ) -> torch.Tensor:
        h = self.ln1(q)
        a, _ = self.attn(h, ctx, ctx, key_padding_mask=ctx_key_padding_mask, need_weights=False)
        x = q + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TextProjector(nn.Module):
    def __init__(self, text_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, d_model)
        self.ln = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.proj(x))  # [B, T, d_model]


class HyperNet(nn.Module):
    def __init__(self, cfg: HyperConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        D = cfg.weight_dim
        self.num_latents = cfg.num_latents
        self.grad_ckpt = getattr(cfg, "grad_ckpt", False)

        self.text_proj = TextProjector(cfg.text_dim, d_model)

        # Module / layer conditioning
        names = list(cfg.module_specs.keys()) if getattr(cfg, "module_specs", None) else []
        self.module_to_idx = {name: i + 1 for i, name in enumerate(names)}
        self.include_emb = cfg.include_emb
        if self.include_emb:
            vocab_size = len(self.module_to_idx) + 1
            self.layer_emb = nn.Embedding(cfg.max_layers, cfg.layer_embed_dim)
            self.module_emb = nn.Embedding(vocab_size, cfg.module_embed_dim)
            self.cond_proj = nn.Linear(cfg.layer_embed_dim + cfg.module_embed_dim, d_model)
            nn.init.xavier_uniform_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

        # Weight encoders (rows and columns of the base weight)
        self.col_enc = nn.Linear(D, d_model)
        self.row_enc = nn.Linear(D, d_model)
        nn.init.xavier_uniform_(self.col_enc.weight)
        nn.init.zeros_(self.col_enc.bias)
        nn.init.xavier_uniform_(self.row_enc.weight)
        nn.init.zeros_(self.row_enc.bias)

        self.latents = nn.Parameter(torch.randn(self.num_latents, d_model))
        self.compressor = CrossAttnBlock(d_model, cfg.n_heads, cfg.dropout)
        self.proc_self = nn.ModuleList([SelfAttnBlock(d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layer)])
        self.proc_cross = nn.ModuleList([CrossAttnBlock(d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layer)])
        self.decompressor = CrossAttnBlock(d_model, cfg.n_heads, cfg.dropout)
        self.head_A = nn.Linear(d_model, cfg.rank)
        self.head_B = nn.Linear(d_model, cfg.rank)

        # Init heads so the initial LoRA update is (near) zero.
        bound = 1 / math.sqrt(D)
        nn.init.zeros_(self.head_A.weight)
        nn.init.uniform_(self.head_A.bias, -bound, bound)
        nn.init.zeros_(self.head_B.weight)
        nn.init.zeros_(self.head_B.bias)
        nn.init.normal_(self.latents, std=0.02)

    @torch.compiler.disable
    def _module_index_tensor(self, module_type, N, device):
        if (module_type is None) or (len(module_type) == 0):
            return torch.zeros(N, dtype=torch.long, device=device)
        idx = [self.module_to_idx.get(name, 0) for name in module_type]
        return torch.tensor(idx, dtype=torch.long, device=device)

    def _expand_cartesian(self, text, base_weight, module_idx_N, layer_index_N, text_key_padding_mask):
        """Expand a [B] text batch and [N] weight batch into their [B * N] cartesian product."""
        B, T, _ = text.shape
        N, D, D2 = base_weight.shape
        text_rep = text.repeat_interleave(N, dim=0)
        weight_rep = base_weight.unsqueeze(0).expand(B, N, D, D).reshape(B * N, D, D)
        mod_rep = module_idx_N.unsqueeze(0).expand(B, N).reshape(B * N)
        layer_rep = layer_index_N.unsqueeze(0).expand(B, N).reshape(B * N) if layer_index_N is not None else None
        kpm_rep = text_key_padding_mask.bool().repeat_interleave(N, dim=0) if text_key_padding_mask is not None else None
        return text_rep, weight_rep, mod_rep, layer_rep, kpm_rep

    def _maybe_checkpoint(self, fn, *args):
        if self.grad_ckpt and self.training:
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
        return fn(*args)

    def forward(
        self,
        text: torch.Tensor,
        base_weight: torch.Tensor,
        module_type: Optional[List[str]] = None,
        layer_index: Optional[torch.LongTensor] = None,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, T, _ = text.shape
        N, D, _ = base_weight.shape
        device, dtype = text.device, text.dtype
        mod_idx_N = self._module_index_tensor(module_type, N, device=device)
        if text_key_padding_mask is not None:
            text_key_padding_mask = text_key_padding_mask.to(device=device, dtype=torch.bool)

        # When text and weights have different batch sizes, take their cartesian product.
        if text.size(0) != base_weight.size(0):
            text_rep, W_rep, mod_rep, layer_rep, kpm_rep = self._expand_cartesian(
                text, base_weight, mod_idx_N, layer_index, text_key_padding_mask
            )
        else:
            text_rep, W_rep, mod_rep, layer_rep, kpm_rep = (
                text, base_weight, mod_idx_N, layer_index, text_key_padding_mask
            )
        BN = text_rep.size(0)

        ctx = self.text_proj(text_rep)               # [BN, T, d_model]
        cols = self.col_enc(W_rep.transpose(1, 2))   # [BN, D, d_model]
        rows = self.row_enc(W_rep)                   # [BN, D, d_model]

        if self.cfg.use_sin_pos:
            pos_D = sinusoidal_positions(D, self.cfg.hidden_dim, device, dtype).unsqueeze(0)
            cols = cols + pos_D
            rows = rows + pos_D

        # Condition the weight tokens on (layer_idx, module_type).
        if self.include_emb:
            e_module = self.module_emb(mod_rep)
            e_layer = (
                self.layer_emb(layer_rep.clamp(min=0, max=self.cfg.max_layers - 1))
                if layer_rep is not None
                else torch.zeros(BN, self.cfg.layer_embed_dim, device=device, dtype=dtype)
            )
            cond_vec = self.cond_proj(torch.cat([e_layer, e_module], dim=-1)).to(dtype)
            cols = cols + cond_vec.unsqueeze(1)
            rows = rows + cond_vec.unsqueeze(1)

        kv_weights = torch.cat([cols, rows], dim=1)            # [BN, 2D, d_model]
        latents = self.latents.unsqueeze(0).expand(BN, -1, -1)  # [BN, n_latents, d_model]

        # Compress the weight tokens into the latents (weights have no padding mask).
        def _compress_fn(q, kv):
            return self.compressor(q, kv, ctx_key_padding_mask=None)
        latents = self._maybe_checkpoint(_compress_fn, latents, kv_weights)

        # Process: alternate latent self-attention and cross-attention to the text context.
        for sb, cb in zip(self.proc_self, self.proc_cross):
            latents = self._maybe_checkpoint(sb, latents)
            if kpm_rep is None:
                def _cb_fn(q, k):
                    return cb(q, k, ctx_key_padding_mask=None)
                latents = self._maybe_checkpoint(_cb_fn, latents, ctx)
            else:
                def _cb_fn(q, k, m):
                    return cb(q, k, ctx_key_padding_mask=m)
                latents = self._maybe_checkpoint(_cb_fn, latents, ctx, kpm_rep)

        # Decompress: weight tokens query the processed latents.
        out_queries = torch.cat([cols, rows], dim=1)

        def _decompress_fn(q, kv):
            return self.decompressor(q, kv, ctx_key_padding_mask=None)
        out_feats = self._maybe_checkpoint(_decompress_fn, out_queries, latents)

        A_feats = out_feats[:, :D, :]
        B_feats = out_feats[:, D:, :]
        Lora_A = self.head_A(A_feats)
        Lora_B = self.head_B(B_feats).transpose(1, 2)
        return {"Lora_A": Lora_A, "Lora_B": Lora_B, "alpha": self.cfg.alpha_scale, "rank": self.cfg.rank}
