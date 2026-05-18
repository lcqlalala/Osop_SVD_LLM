#coding:utf8
import argparse
import copy
import heapq
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from evaluater import eff_eval, ppl_eval
from utils.data_utils import get_calib_train_data
from utils.model_utils import find_layers, get_model_from_huggingface, get_model_from_local


current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)


def get_transformer_layers(model_name: str, model):
    if "opt" in model_name:
        return model.model.decoder.layers
    return model.model.layers


def patch_svd_layer_indices(model_name: str, model):
    if "llama" not in model_name and "vicuna" not in model_name:
        return
    for layer_idx, layer in enumerate(get_transformer_layers(model_name, model)):
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, SVD_LlamaAttention):
            layer.self_attn.layer_idx = layer_idx


def move_embeddings(model_name: str, model, device):
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(device)
    else:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)


def move_embeddings_cpu(model_name: str, model):
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()


def target_rank(weight: torch.Tensor, keep_ratio: float) -> int:
    rows, columns = weight.shape
    rank = int(rows * columns * keep_ratio / (rows + columns))
    return max(1, rank)


def component_rank(model_name: str, model, name: str, weight: torch.Tensor, keep_ratio: float) -> int:
    if ("llama" in model_name or "vicuna" in model_name) and any(
        proj in name for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
    ):
        return max(1, int(model.config.hidden_size * keep_ratio / 2))
    if "mistral" in model_name and any(proj in name for proj in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return max(1, int(model.config.hidden_size * keep_ratio / 2))
    return target_rank(weight, keep_ratio)


def pad_factors(a: torch.Tensor, b: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    current_rank = a.shape[1]
    if current_rank == rank:
        return a, b
    if current_rank > rank:
        return a[:, :rank], b[:rank, :]
    pad_rank = rank - current_rank
    a_pad = torch.zeros((a.shape[0], pad_rank), dtype=a.dtype, device=a.device)
    b_pad = torch.zeros((pad_rank, b.shape[1]), dtype=b.dtype, device=b.device)
    return torch.cat((a, a_pad), dim=1), torch.cat((b, b_pad), dim=0)


def flatten_activations(inp: torch.Tensor) -> torch.Tensor:
    if inp.dim() == 2:
        return inp
    return inp.reshape(-1, inp.shape[-1])


def cat_optional_tensors(tensors):
    if all(tensor is None for tensor in tensors):
        return None
    if any(tensor is None for tensor in tensors):
        raise RuntimeError("Mixed None and Tensor values were captured for a replay argument.")
    return torch.cat(tensors, dim=0)


def load_gamma_source(path: Optional[str]):
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


def lookup_external_gamma(gamma_source, layer_idx: int, name: str):
    if gamma_source is None:
        return None
    for layer_key in (layer_idx, str(layer_idx)):
        if isinstance(gamma_source, dict) and layer_key in gamma_source:
            layer_value = gamma_source[layer_key]
            if isinstance(layer_value, dict):
                return layer_value.get(name)
    full_name = f"{layer_idx}.{name}"
    if isinstance(gamma_source, dict):
        if full_name in gamma_source:
            return gamma_source[full_name]
        if name in gamma_source:
            return gamma_source[name]
    return None


def build_gamma_diag(
    weight: torch.Tensor,
    gamma_mode: str = "ones",
    damping: float = 1e-6,
    external_gamma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if external_gamma is not None:
        gamma = external_gamma.detach().float().cpu().flatten()
    elif gamma_mode == "ones":
        gamma = torch.ones(weight.shape[0], dtype=torch.float32)
    elif gamma_mode == "row_norm":
        gamma = weight.detach().float().cpu().pow(2).mean(dim=1)
    elif gamma_mode == "row_abs_mean":
        gamma = weight.detach().float().cpu().abs().mean(dim=1)
    else:
        raise ValueError(f"Unknown gamma mode: {gamma_mode}")
    if gamma.numel() != weight.shape[0]:
        raise ValueError(f"Gamma length {gamma.numel()} does not match d_out {weight.shape[0]}")
    return torch.clamp(gamma, min=damping)


def parse_target_patterns(patterns: Optional[str]):
    if patterns is None or patterns.strip() == "" or patterns.strip().lower() == "all":
        return None
    return [pattern.strip() for pattern in patterns.split(",") if pattern.strip()]


def name_matches_patterns(name: str, patterns) -> bool:
    if patterns is None:
        return True
    return any(pattern in name for pattern in patterns)


def compressed_v_name(name: str) -> str:
    replacements = (
        ("q_proj", "q_v_proj"),
        ("k_proj", "k_v_proj"),
        ("v_proj", "v_v_proj"),
        ("o_proj", "o_v_proj"),
        ("out_proj", "out_v_proj"),
        ("gate_proj", "gate_v_proj"),
        ("down_proj", "down_v_proj"),
        ("up_proj", "up_v_proj"),
        ("fc1", "fc1_v_proj"),
        ("fc2", "fc2_v_proj"),
    )
    for src, dst in replacements:
        if name.endswith(src):
            return name[: -len(src)] + dst
    raise KeyError(f"Cannot map original Linear name to compressed v-proj name: {name}")


def compressed_u_name(name: str) -> str:
    replacements = (
        ("q_proj", "q_u_proj"),
        ("k_proj", "k_u_proj"),
        ("v_proj", "v_u_proj"),
        ("o_proj", "o_u_proj"),
        ("out_proj", "out_u_proj"),
        ("gate_proj", "gate_u_proj"),
        ("down_proj", "down_u_proj"),
        ("up_proj", "up_u_proj"),
        ("fc1", "fc1_u_proj"),
        ("fc2", "fc2_u_proj"),
    )
    for src, dst in replacements:
        if name.endswith(src):
            return name[: -len(src)] + dst
    raise KeyError(f"Cannot map original Linear name to compressed u-proj name: {name}")


class OSOPAccumulator:
    """Streams either output-space or input-space Gram matrices for one Linear layer."""

    def __init__(
        self,
        layer: nn.Linear,
        rank: int,
        gamma_diag: torch.Tensor,
        method: str,
        damping: float,
        accum_dtype: torch.dtype = torch.float32,
    ):
        self.layer = layer
        self.rank = rank
        self.gamma_diag = gamma_diag.cpu().float()
        self.gamma_sqrt = torch.sqrt(self.gamma_diag)
        self.method = method
        self.damping = damping
        self.accum_dtype = accum_dtype
        self.nsamples = 0

        rows, columns = layer.weight.shape
        if method == "osop":
            self.gram = torch.zeros((rows, rows), dtype=accum_dtype, device="cpu")
        elif method == "whitening":
            self.gram = torch.zeros((columns, columns), dtype=accum_dtype, device="cpu")
        else:
            raise ValueError(f"Unknown OSOP accumulator method: {method}")

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        x = flatten_activations(inp.detach()).float()
        self.nsamples += x.shape[0]
        if self.method == "osop":
            weight = self.layer.weight.detach().float()
            y = x.matmul(weight.t())
            gamma_sqrt = self.gamma_sqrt.to(y.device, dtype=y.dtype)
            y = y * gamma_sqrt.unsqueeze(0)
            self.gram += y.t().matmul(y).to("cpu", dtype=self.accum_dtype)
        else:
            self.gram += x.t().matmul(x).to("cpu", dtype=self.accum_dtype)

    @torch.no_grad()
    def eigenvalues_desc(self) -> torch.Tensor:
        if self.method != "osop":
            raise RuntimeError("OSOP eigen spectrum is only defined for output-space accumulation.")
        if self.nsamples == 0:
            raise RuntimeError("No calibration samples were accumulated.")
        gram = (self.gram.float() / float(self.nsamples)).contiguous()
        gram = 0.5 * (gram + gram.t())
        eigvals = torch.linalg.eigvalsh(gram)
        eigvals = torch.clamp(eigvals, min=0)
        return torch.flip(eigvals, dims=[0]).cpu()

    @torch.no_grad()
    def factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.nsamples == 0:
            raise RuntimeError("No calibration samples were accumulated.")

        weight = self.layer.weight.detach().float().cpu()
        rank = min(self.rank, weight.shape[0], weight.shape[1])
        gamma_sqrt = self.gamma_sqrt.float()
        inv_gamma_sqrt = torch.rsqrt(torch.clamp(self.gamma_diag.float(), min=self.damping))

        if self.method == "osop":
            gram = (self.gram.float() / float(self.nsamples)).contiguous()
            gram = 0.5 * (gram + gram.t())
            _, eigvecs = torch.linalg.eigh(gram)
            u_k = eigvecs[:, -rank:].contiguous()
            a = inv_gamma_sqrt.unsqueeze(1) * u_k
            b = u_k.t().matmul(gamma_sqrt.unsqueeze(1) * weight)
            return pad_factors(a, b, self.rank)

        gram = (self.gram.float() / float(self.nsamples)).contiguous()
        eye = torch.eye(gram.shape[0], dtype=gram.dtype)
        gram = 0.5 * (gram + gram.t()) + self.damping * eye
        try:
            scaling = torch.linalg.cholesky(gram)
        except RuntimeError:
            eig_min = torch.linalg.eigvalsh(gram)[0].item()
            gram = gram + (-eig_min + self.damping) * eye
            scaling = torch.linalg.cholesky(gram)
        scaling_inv = torch.linalg.inv(scaling)
        weighted = gamma_sqrt.unsqueeze(1) * weight
        u, s, vt = torch.linalg.svd(weighted.matmul(scaling), full_matrices=False)
        u_k = u[:, :rank]
        s_k = s[:rank]
        vt_k = vt[:rank, :]
        sqrt_sigma = torch.diag(torch.sqrt(s_k))
        a = inv_gamma_sqrt.unsqueeze(1) * u_k.matmul(sqrt_sigma)
        b = sqrt_sigma.matmul(vt_k).matmul(scaling_inv)
        return pad_factors(a, b, self.rank)


class LowRankRefitAccumulator:
    """Least-squares refit of A with B fixed: min_A ||Y - (X B^T) A^T||_F^2."""

    def __init__(self, b: torch.Tensor, out_features: int, damping: float, accum_dtype: torch.dtype = torch.float32):
        self.b = b.detach().float()
        self.rank = b.shape[0]
        self.out_features = out_features
        self.damping = damping
        self.accum_dtype = accum_dtype
        self.hth = torch.zeros((self.rank, self.rank), dtype=accum_dtype, device="cpu")
        self.hty = torch.zeros((self.rank, out_features), dtype=accum_dtype, device="cpu")
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        x = flatten_activations(inp.detach()).float()
        y = flatten_activations(out.detach()).float()
        b = self.b.to(x.device, dtype=x.dtype)
        h = x.matmul(b.t())
        self.hth += h.t().matmul(h).to("cpu", dtype=self.accum_dtype)
        self.hty += h.t().matmul(y).to("cpu", dtype=self.accum_dtype)
        self.nsamples += x.shape[0]

    @torch.no_grad()
    def solve_a(self) -> torch.Tensor:
        if self.nsamples == 0:
            raise RuntimeError("No calibration samples were accumulated for local refit.")
        hth = self.hth.float()
        hty = self.hty.float()
        eye = torch.eye(self.rank, dtype=hth.dtype)
        try:
            a_t = torch.linalg.solve(hth + self.damping * eye, hty)
        except RuntimeError:
            a_t = torch.linalg.lstsq(hth + self.damping * eye, hty).solution
        return a_t.t().contiguous()


def choose_method(weight: torch.Tensor, mode: str, osop_dim_threshold: float = 1.0) -> str:
    rows, columns = weight.shape
    if mode == "osop":
        return "osop"
    if mode == "whitening":
        return "whitening"
    if mode == "hybrid":
        return "osop" if rows <= columns * osop_dim_threshold else "whitening"
    raise ValueError(f"Unknown compression mode: {mode}")


def build_svd_modules(model_name: str, model, layer, keep_ratio: float, layer_idx: int = 0, rank_config=None):
    if "llama" in model_name or "vicuna" in model_name:
        svd_attn = SVD_LlamaAttention(config=model.config, ratio=keep_ratio, rank_config=rank_config)
        svd_attn.layer_idx = layer_idx
        svd_mlp = SVD_LlamaMLP(
            hidden_size=layer.hidden_size,
            intermediate_size=model.config.intermediate_size,
            hidden_act=model.config.hidden_act,
            ratio=keep_ratio,
            rank_config=rank_config,
        )
        return svd_attn, svd_mlp, None
    if "mistral" in model_name:
        return (
            SVD_MistralAttention(config=model.config, ratio=keep_ratio),
            SVD_MistralMLP(config=model.config, ratio=keep_ratio),
            None,
        )
    if "opt" in model_name:
        return None, None, SVDOPTDecoderLayer(model.config, ratio=keep_ratio)
    raise NotImplementedError(f"Unsupported model family for OSOP compression: {model_name}")


def assign_factor(model_name: str, layer, svd_attn, svd_mlp, svd_decoder, name: str, a, b, dtype):
    a = a.to(dtype)
    b = b.to(dtype)
    if "opt" in model_name:
        if "q_proj" in name:
            svd_decoder.self_attn.q_u_proj.weight.data = a
            svd_decoder.self_attn.q_v_proj.weight.data = b
            svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
        elif "k_proj" in name:
            svd_decoder.self_attn.k_u_proj.weight.data = a
            svd_decoder.self_attn.k_v_proj.weight.data = b
            svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
        elif "v_proj" in name:
            svd_decoder.self_attn.v_u_proj.weight.data = a
            svd_decoder.self_attn.v_v_proj.weight.data = b
            svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
        elif "out_proj" in name:
            svd_decoder.self_attn.out_u_proj.weight.data = a
            svd_decoder.self_attn.out_v_proj.weight.data = b
            svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
        elif "fc1" in name:
            svd_decoder.fc1_u_proj.weight.data = a
            svd_decoder.fc1_v_proj.weight.data = b
            svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
        elif "fc2" in name:
            svd_decoder.fc2_u_proj.weight.data = a
            svd_decoder.fc2_v_proj.weight.data = b
            svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
        return

    if "q_proj" in name:
        svd_attn.q_u_proj.weight.data = a
        svd_attn.q_v_proj.weight.data = b
    elif "k_proj" in name:
        svd_attn.k_u_proj.weight.data = a
        svd_attn.k_v_proj.weight.data = b
    elif "v_proj" in name:
        svd_attn.v_u_proj.weight.data = a
        svd_attn.v_v_proj.weight.data = b
    elif "o_proj" in name:
        svd_attn.o_u_proj.weight.data = a
        svd_attn.o_v_proj.weight.data = b
        layer.self_attn = svd_attn
    elif "gate_proj" in name:
        svd_mlp.gate_u_proj.weight.data = a
        svd_mlp.gate_v_proj.weight.data = b
    elif "down_proj" in name:
        svd_mlp.down_u_proj.weight.data = a
        svd_mlp.down_v_proj.weight.data = b
    elif "up_proj" in name:
        svd_mlp.up_u_proj.weight.data = a
        svd_mlp.up_v_proj.weight.data = b
        layer.mlp = svd_mlp


def module_param_cost(weight: torch.Tensor) -> int:
    return int(weight.shape[0] + weight.shape[1])


def allocate_energy_guided_ranks(stats, min_rank: int = 1):
    total_budget = 0
    plan = {}
    heap = []
    used_budget = 0
    stats_by_key = {stat["key"]: stat for stat in stats}

    for stat in stats:
        key = stat["key"]
        cost = stat["cost"]
        base_rank = stat["base_rank"]
        max_rank = stat["max_rank"]
        eigvals = stat["eigvals"]
        total_budget += base_rank * cost
        rank = min(max(min_rank, 1), max_rank)
        plan[key] = rank
        used_budget += rank * cost
        if rank < max_rank and rank < eigvals.numel():
            gain = float(eigvals[rank].item())
            heapq.heappush(heap, (-(gain / cost), key, gain, cost))

    while heap:
        neg_gain_per_cost, key, gain, cost = heapq.heappop(heap)
        if used_budget + cost > total_budget:
            continue
        stat = stats_by_key[key]
        rank = plan[key]
        if rank >= stat["max_rank"]:
            continue
        used_budget += cost
        rank += 1
        plan[key] = rank
        if rank < stat["max_rank"] and rank < stat["eigvals"].numel():
            next_gain = float(stat["eigvals"][rank].item())
            heapq.heappush(heap, (-(next_gain / cost), key, next_gain, cost))

    return plan, total_budget, used_budget


@torch.no_grad()
def collect_osop_rank_stats(
    model_name: str,
    model,
    layers,
    inps: torch.Tensor,
    attention_masks,
    position_ids,
    keep_ratio: float,
    dev: str,
    gamma_mode: str,
    gamma_source,
    damping: float,
    accum_dtype: torch.dtype,
):
    print("Collecting OSOP spectra for energy-guided rank reallocation...")
    stats = []
    rank_inps = inps
    for layer_idx in tqdm(range(len(layers))):
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        accumulators = {}

        for name, module in subset.items():
            base_rank = component_rank(model_name, model, name, module.weight, keep_ratio)
            external_gamma = lookup_external_gamma(gamma_source, layer_idx, name)
            gamma_diag = build_gamma_diag(
                module.weight,
                gamma_mode=gamma_mode,
                damping=damping,
                external_gamma=external_gamma,
            )
            accumulators[name] = OSOPAccumulator(
                module,
                rank=base_rank,
                gamma_diag=gamma_diag,
                method="osop",
                damping=damping,
                accum_dtype=accum_dtype,
            )

        handles = []
        for name, module in subset.items():
            def add_rank_stat_batch(module_, inp, out, layer_name=name):
                del module_, out
                accumulators[layer_name].add_batch(inp[0])
            handles.append(module.register_forward_hook(add_rank_stat_batch))

        outs = []
        for sample_idx in range(rank_inps.shape[0]):
            inp = rank_inps[sample_idx:sample_idx + 1].to(dev)
            attn = None if attention_masks is None else attention_masks[sample_idx:sample_idx + 1].to(dev)
            pos = None if position_ids is None else position_ids[sample_idx:sample_idx + 1].to(dev)
            outs.append(run_decoder_layer(model_name, layer, inp, attn, pos).detach().cpu())

        for handle in handles:
            handle.remove()

        for name, module in subset.items():
            eigvals = accumulators[name].eigenvalues_desc()
            base_rank = min(component_rank(model_name, model, name, module.weight, keep_ratio), module.weight.shape[0], module.weight.shape[1])
            max_rank = min(module.weight.shape[0], module.weight.shape[1], eigvals.numel())
            stats.append({
                "key": (layer_idx, name),
                "layer_idx": layer_idx,
                "name": name,
                "base_rank": base_rank,
                "max_rank": max_rank,
                "cost": module_param_cost(module.weight),
                "eigvals": eigvals,
            })

        layers[layer_idx] = layer.cpu()
        rank_inps = torch.cat(outs, dim=0)
        del outs, accumulators, layer
        torch.cuda.empty_cache()

    plan, total_budget, used_budget = allocate_energy_guided_ranks(stats)
    print(f"Rank reallocation budget: base={total_budget}, allocated={used_budget}")
    return plan


@torch.no_grad()
def run_decoder_layer(model_name: str, layer, inp, attention_mask, position_ids):
    if "opt" in model_name:
        return layer(inp, attention_mask=attention_mask)[0]
    return layer(inp, attention_mask=attention_mask, position_ids=position_ids)[0]


@torch.no_grad()
def collect_first_layer_inputs(model_name: str, model, calib_loader, dev):
    layers = get_transformer_layers(model_name, model)
    move_embeddings(model_name, model, dev)
    layers[0] = layers[0].to(dev)

    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            attention_mask = kwargs.get("attention_mask")
            attention_masks.append(None if attention_mask is None else attention_mask.detach().cpu())
            if "opt" not in model_name:
                pos = kwargs.get("position_ids")
                position_ids.append(None if pos is None else pos.detach().cpu())
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module.cpu()
    move_embeddings_cpu(model_name, model)
    torch.cuda.empty_cache()

    inps = torch.cat(inps, dim=0)
    attention_masks = cat_optional_tensors(attention_masks)
    if "opt" in model_name:
        position_ids = None
    else:
        position_ids = cat_optional_tensors(position_ids)
    return inps, attention_masks, position_ids


@torch.no_grad()
def osop_compress(
    model_name: str,
    model,
    calib_loader,
    keep_ratio: float,
    dev: str = "cuda",
    mode: str = "hybrid",
    gamma_mode: str = "ones",
    gamma_source=None,
    damping: float = 1e-6,
    accum_dtype: torch.dtype = torch.float32,
    osop_dim_threshold: float = 1.0,
    rank_realloc: bool = False,
    propagate_compressed_outputs: bool = False,
    local_update: bool = False,
    teacher_update: bool = False,
    teacher_update_targets: Optional[str] = None,
    teacher_update_alpha: float = 1.0,
    teacher_update_norm_clip: float = 0.0,
    local_update_damping: float = 1e-4,
):
    print("Collecting calibration activations for OSOP...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_transformer_layers(model_name, model)
    inps, attention_masks, position_ids = collect_first_layer_inputs(model_name, model, calib_loader, dev)
    teacher_inps = inps if teacher_update else None
    teacher_target_patterns = parse_target_patterns(teacher_update_targets)
    dtype = next(iter(model.parameters())).dtype
    rank_plan = {}
    if rank_realloc:
        if mode != "osop":
            print("Warning: rank reallocation is designed for OSOP; non-OSOP fallback modules will still use reallocated ranks.")
        rank_plan = collect_osop_rank_stats(
            model_name=model_name,
            model=model,
            layers=layers,
            inps=inps,
            attention_masks=attention_masks,
            position_ids=position_ids,
            keep_ratio=keep_ratio,
            dev=dev,
            gamma_mode=gamma_mode,
            gamma_source=gamma_source,
            damping=damping,
            accum_dtype=accum_dtype,
        )

    print("Start OSOP compression...")
    for layer_idx in tqdm(range(len(layers))):
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        accumulators: Dict[str, OSOPAccumulator] = {}
        method_counts = {"osop": 0, "whitening": 0}
        layer_rank_config = {}

        for name, module in subset.items():
            method = choose_method(module.weight, mode, osop_dim_threshold=osop_dim_threshold)
            method_counts[method] += 1
            rank = rank_plan.get(
                (layer_idx, name),
                component_rank(model_name, model, name, module.weight, keep_ratio),
            )
            layer_rank_config[name] = rank
            external_gamma = lookup_external_gamma(gamma_source, layer_idx, name)
            gamma_diag = build_gamma_diag(
                module.weight,
                gamma_mode=gamma_mode,
                damping=damping,
                external_gamma=external_gamma,
            )
            accumulators[name] = OSOPAccumulator(
                module,
                rank=rank,
                gamma_diag=gamma_diag,
                method=method,
                damping=damping,
                accum_dtype=accum_dtype,
            )

        handles = []
        for name, module in subset.items():
            def add_batch_hook(module_, inp, out, layer_name=name):
                del module_, out
                accumulators[layer_name].add_batch(inp[0])
            handles.append(module.register_forward_hook(add_batch_hook))

        outs = []
        for sample_idx in range(inps.shape[0]):
            inp = inps[sample_idx:sample_idx + 1].to(dev)
            attn = None if attention_masks is None else attention_masks[sample_idx:sample_idx + 1].to(dev)
            if "opt" in model_name:
                out = layer(inp, attention_mask=attn)[0]
            else:
                pos = None if position_ids is None else position_ids[sample_idx:sample_idx + 1].to(dev)
                out = layer(inp, attention_mask=attn, position_ids=pos)[0]
            outs.append(out.detach().cpu())

        for handle in handles:
            handle.remove()

        svd_attn, svd_mlp, svd_decoder = build_svd_modules(
            model_name,
            model,
            layer,
            keep_ratio,
            layer_idx,
            rank_config=layer_rank_config,
        )
        factors = {}
        for name, accumulator in accumulators.items():
            a, b = accumulator.factors()
            factors[name] = [a, b]

        if local_update:
            refitters: Dict[str, LowRankRefitAccumulator] = {}
            for name, module in subset.items():
                refitters[name] = LowRankRefitAccumulator(
                    b=factors[name][1],
                    out_features=module.weight.shape[0],
                    damping=local_update_damping,
                    accum_dtype=accum_dtype,
                )

            refit_handles = []
            for name, module in subset.items():
                def add_refit_batch(module_, inp, out, layer_name=name):
                    del module_
                    refitters[layer_name].add_batch(inp[0], out)
                refit_handles.append(module.register_forward_hook(add_refit_batch))

            for sample_idx in range(inps.shape[0]):
                inp = inps[sample_idx:sample_idx + 1].to(dev)
                attn = None if attention_masks is None else attention_masks[sample_idx:sample_idx + 1].to(dev)
                pos = None if position_ids is None else position_ids[sample_idx:sample_idx + 1].to(dev)
                run_decoder_layer(model_name, layer, inp, attn, pos)

            for handle in refit_handles:
                handle.remove()

            for name, refitter in refitters.items():
                factors[name][0] = refitter.solve_a()

        teacher_layer = copy.deepcopy(layer).to(dev) if teacher_update else None

        for name, (a, b) in factors.items():
            assign_factor(model_name, layer, svd_attn, svd_mlp, svd_decoder, name, a, b, dtype)

        if "opt" in model_name:
            svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
            svd_decoder.final_layer_norm = layer.final_layer_norm
            layers[layer_idx] = svd_decoder.cpu()
        else:
            layers[layer_idx] = layer.cpu()

        if teacher_update:
            compressed_layer = layers[layer_idx].to(dev)
            teacher_subset = find_layers(teacher_layer)
            compressed_subset = find_layers(compressed_layer)
            target_names = [
                name for name in subset
                if name_matches_patterns(name, teacher_target_patterns)
            ]
            refitters: Dict[str, LowRankRefitAccumulator] = {}
            for name in target_names:
                v_name = compressed_v_name(name)
                refitters[name] = LowRankRefitAccumulator(
                    b=compressed_subset[v_name].weight.detach().float().cpu(),
                    out_features=teacher_subset[name].weight.shape[0],
                    damping=local_update_damping,
                    accum_dtype=accum_dtype,
                )

            teacher_outputs = {}
            teacher_outs = []
            teacher_handles = []
            compressed_handles = []

            for name in target_names:
                def save_teacher_output(module_, inp, out, layer_name=name):
                    del module_, inp
                    teacher_outputs[layer_name] = out.detach()
                teacher_handles.append(teacher_subset[name].register_forward_hook(save_teacher_output))

                v_name = compressed_v_name(name)
                def add_compressed_input(module_, inp, out, layer_name=name):
                    del module_, out
                    refitters[layer_name].add_batch(inp[0], teacher_outputs[layer_name])
                compressed_handles.append(compressed_subset[v_name].register_forward_hook(add_compressed_input))

            for sample_idx in range(inps.shape[0]):
                attn = None if attention_masks is None else attention_masks[sample_idx:sample_idx + 1].to(dev)
                pos = None if position_ids is None else position_ids[sample_idx:sample_idx + 1].to(dev)

                teacher_outputs.clear()
                teacher_inp = teacher_inps[sample_idx:sample_idx + 1].to(dev)
                teacher_out = run_decoder_layer(model_name, teacher_layer, teacher_inp, attn, pos)
                teacher_outs.append(teacher_out.detach().cpu())

                inp = inps[sample_idx:sample_idx + 1].to(dev)
                run_decoder_layer(model_name, compressed_layer, inp, attn, pos)

            for handle in teacher_handles + compressed_handles:
                handle.remove()

            compressed_subset = find_layers(compressed_layer)
            for name, refitter in refitters.items():
                u_name = compressed_u_name(name)
                base_a = compressed_subset[u_name].weight.data.float().cpu()
                teacher_a = refitter.solve_a()
                if teacher_update_norm_clip > 0:
                    base_norm = torch.linalg.norm(base_a)
                    teacher_norm = torch.linalg.norm(teacher_a)
                    max_norm = teacher_update_norm_clip * base_norm
                    if teacher_norm > max_norm:
                        teacher_a = teacher_a * (max_norm / (teacher_norm + 1e-12))
                blended_a = (1 - teacher_update_alpha) * base_a + teacher_update_alpha * teacher_a
                compressed_subset[u_name].weight.data = blended_a.to(dtype)

            teacher_inps = torch.cat(teacher_outs, dim=0)
            layers[layer_idx] = compressed_layer.cpu()
            teacher_layer = teacher_layer.cpu()
            del teacher_outputs, teacher_outs, teacher_layer, compressed_layer, refitters

        if local_update or teacher_update or propagate_compressed_outputs:
            compressed_layer = layers[layer_idx].to(dev)
            compressed_outs = []
            for sample_idx in range(inps.shape[0]):
                inp = inps[sample_idx:sample_idx + 1].to(dev)
                attn = None if attention_masks is None else attention_masks[sample_idx:sample_idx + 1].to(dev)
                pos = None if position_ids is None else position_ids[sample_idx:sample_idx + 1].to(dev)
                compressed_out = run_decoder_layer(model_name, compressed_layer, inp, attn, pos)
                compressed_outs.append(compressed_out.detach().cpu())
            layers[layer_idx] = compressed_layer.cpu()
            inps = torch.cat(compressed_outs, dim=0)
            del compressed_outs, compressed_layer
        else:
            inps = torch.cat(outs, dim=0)
        del outs, accumulators, layer, factors
        torch.cuda.empty_cache()

        print(
            f"Layer {layer_idx}: OSOP={method_counts['osop']}, "
            f"input-whitening fallback={method_counts['whitening']}"
        )

    model.config.use_cache = use_cache
    return model


def parse_accum_dtype(name: str):
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("--accum_dtype must be float32 or float64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jeffwan/llama-7b-hf")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=0.2, help="Compression ratio. 0.2 means keep roughly 80% in this codebase convention.")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--DEV", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["osop", "whitening", "hybrid"])
    parser.add_argument(
        "--osop_dim_threshold",
        type=float,
        default=1.0,
        help="In hybrid mode, use OSOP only when d_out <= threshold * d_in. Try 0.75 to keep OSOP mainly for down_proj.",
    )
    parser.add_argument(
        "--rank_realloc",
        action="store_true",
        help="Use Energy-Guided OSOP rank reallocation under the same global parameter budget.",
    )
    parser.add_argument("--gamma_mode", type=str, default="ones", choices=["ones", "row_norm", "row_abs_mean"])
    parser.add_argument("--gamma_path", type=str, default=None, help="Optional torch file: {layer_idx: {linear_name: gamma_diag}}.")
    parser.add_argument("--damping", type=float, default=1e-6)
    parser.add_argument("--accum_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--propagate_compressed_outputs",
        action="store_true",
        help="After each layer is compressed, feed its compressed outputs to calibrate later layers.",
    )
    parser.add_argument("--local_update", action="store_true", help="Refit A with B fixed on calibration activations and propagate compressed layer outputs.")
    parser.add_argument(
        "--teacher_update",
        action="store_true",
        help="Refit A with compressed-prefix module inputs and teacher-trajectory module outputs.",
    )
    parser.add_argument(
        "--teacher_update_targets",
        type=str,
        default="all",
        help="Comma-separated substrings of Linear names to teacher-update, e.g. 'o_proj,down_proj'. Use 'all' for every Linear.",
    )
    parser.add_argument("--teacher_update_alpha", type=float, default=1.0, help="Blend teacher-refit A into OSOP A. Smaller values are safer, e.g. 0.05-0.2.")
    parser.add_argument("--teacher_update_norm_clip", type=float, default=0.0, help="If >0, cap ||A_teacher|| to this multiple of ||A_osop|| before blending.")
    parser.add_argument("--local_update_damping", type=float, default=1e-4)
    parser.add_argument("--step", type=int, default=1, help="1: OSOP compress, 4: PPL eval, 5: efficiency eval")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gen_seq_len", type=int, default=1024)

    args = parser.parse_args()
    keep_ratio = 1 - args.ratio

    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        calib_loader = get_calib_train_data(
            args.dataset,
            tokenizer,
            args.nsamples,
            seqlen=args.model_seq_len,
            seed=args.seed,
        )
        gamma_source = load_gamma_source(args.gamma_path)
        osop_compress(
            model_name=args.model,
            model=model,
            calib_loader=calib_loader,
            keep_ratio=keep_ratio,
            dev=args.DEV,
            mode=args.mode,
            gamma_mode=args.gamma_mode,
            gamma_source=gamma_source,
            damping=args.damping,
            accum_dtype=parse_accum_dtype(args.accum_dtype),
            osop_dim_threshold=args.osop_dim_threshold,
            rank_realloc=args.rank_realloc,
            propagate_compressed_outputs=args.propagate_compressed_outputs,
            local_update=args.local_update,
            teacher_update=args.teacher_update,
            teacher_update_targets=args.teacher_update_targets,
            teacher_update_alpha=args.teacher_update_alpha,
            teacher_update_norm_clip=args.teacher_update_norm_clip,
            local_update_damping=args.local_update_damping,
        )
        patch_svd_layer_indices(args.model, model)
        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
            extra_tags = []
            if args.rank_realloc:
                extra_tags.append("rankrealloc")
            if args.propagate_compressed_outputs:
                extra_tags.append("prop")
            if args.local_update:
                extra_tags.append("local")
            if args.teacher_update:
                extra_tags.append("teacher")
            tag = "" if not extra_tags else "_" + "_".join(extra_tags)
            save_name = (
                args.model.replace("/", "_").replace("-", "_")
                + f"_osop_{args.mode}_{keep_ratio}{tag}.pt"
            )
            torch.save({"model": model, "tokenizer": tokenizer}, os.path.join(args.save_path, save_name))
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
        patch_svd_layer_indices(args.model, model)
        model.eval()
        model = model.float().to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=["wikitext2"], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
