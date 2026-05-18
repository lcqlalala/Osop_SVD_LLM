import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig
try:
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding as OfficialLlamaRotaryEmbedding,
        apply_rotary_pos_emb as official_apply_rotary_pos_emb,
        repeat_kv as official_repeat_kv,
    )
except ImportError:
    OfficialLlamaRotaryEmbedding = None
    official_apply_rotary_pos_emb = None
    official_repeat_kv = None

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def rank_from_config(rank_config, name, default):
    if rank_config is None:
        return default
    if name in rank_config:
        return rank_config[name]
    for key, value in rank_config.items():
        if key.endswith(name):
            return value
    return default

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv_fallback(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        cos = cos[:, :, : q.shape[-2], :]
        sin = sin[:, :, : q.shape[-2], :]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_compat(q, k, cos, sin, position_ids=None):
    if official_apply_rotary_pos_emb is not None:
        return official_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    return apply_rotary_pos_emb(q, k, cos, sin, position_ids)


def make_causal_mask(q_len, kv_seq_len, dtype, device):
    min_dtype = torch.finfo(dtype).min
    past_len = kv_seq_len - q_len
    query_positions = torch.arange(q_len, device=device)[:, None]
    key_positions = torch.arange(kv_seq_len, device=device)[None, :]
    allowed = key_positions <= query_positions + past_len
    mask = torch.full((q_len, kv_seq_len), min_dtype, dtype=dtype, device=device)
    mask = mask.masked_fill(allowed, 0)
    return mask[None, None, :, :]


class SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ratio=1,
        rank_config=None,
    ):
        super().__init__()
        self.ratio = ratio
        low_rank = int(intermediate_size * hidden_size * self.ratio / (intermediate_size + hidden_size))
        gate_rank = rank_from_config(rank_config, "gate_proj", low_rank)
        down_rank = rank_from_config(rank_config, "down_proj", low_rank)
        up_rank = rank_from_config(rank_config, "up_proj", low_rank)
        self.gate_u_proj = nn.Linear(gate_rank, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, gate_rank, bias=False)
        
        self.down_u_proj = nn.Linear(down_rank, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, down_rank, bias=False)
        
        self.up_u_proj = nn.Linear(up_rank, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, up_rank, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))


class SVD_LlamaAttention(nn.Module):
    """LLaMA attention with low-rank projections.

    The forward path intentionally mirrors the official Transformers LLaMA
    attention implementation; only q/k/v/o projections are replaced by SVD
    factorized projections.
    """

    def __init__(self, config: LlamaConfig, ratio=1, rank_config=None):
        super().__init__()
        self.config = config
        self.layer_idx = None
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.ratio = ratio # 1 means no truncate, just keep normal attn

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        low_rank = int(self.hidden_size * self.ratio/2)
        q_rank = rank_from_config(rank_config, "q_proj", low_rank)
        k_rank = rank_from_config(rank_config, "k_proj", low_rank)
        v_rank = rank_from_config(rank_config, "v_proj", low_rank)
        o_rank = rank_from_config(rank_config, "o_proj", low_rank)
        self.q_u_proj = nn.Linear(q_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, q_rank, bias=False)

        self.k_u_proj = nn.Linear(k_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, k_rank, bias=False)

        self.v_u_proj = nn.Linear(v_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, v_rank, bias=False)

        self.o_u_proj = nn.Linear(o_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, o_rank, bias=False)

        if OfficialLlamaRotaryEmbedding is not None:
            try:
                self.rotary_emb = OfficialLlamaRotaryEmbedding(config=config)
            except TypeError:
                self.rotary_emb = OfficialLlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _rotary_embedding(self, value_states, position_ids, kv_seq_len):
        try:
            return self.rotary_emb(value_states, position_ids)
        except TypeError:
            return self.rotary_emb(value_states, seq_len=kv_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        del kwargs
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        value_states = self.v_u_proj(self.v_v_proj(hidden_states))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not hasattr(past_key_value, "update"):
            kv_seq_len += past_key_value[0].shape[-2]
        if position_embeddings is None:
            cos, sin = self._rotary_embedding(value_states, position_ids, kv_seq_len)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_compat(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        repeat_kv = official_repeat_kv or repeat_kv_fallback
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, key_states.shape[-2]):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, key_states.shape[-2])}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is None:
            attention_mask = make_causal_mask(q_len, key_states.shape[-2], attn_weights.dtype, attn_weights.device)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            if causal_mask.size(0) == 1 and bsz != 1:
                causal_mask = causal_mask.expand(bsz, -1, -1, -1)
            if causal_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, key_states.shape[-2])}, but is {causal_mask.size()}"
                )
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value if use_cache else None
    
