#coding:utf8
import argparse
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


def move_embeddings(model_name: str, model, device):
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(device)
    else:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)


def move_embeddings_cpu(model_name: str, model):
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()


def target_rank(weight: torch.Tensor, keep_ratio: float) -> int:
    rows, columns = weight.shape
    rank = int(rows * columns * keep_ratio / (rows + columns))
    return max(1, rank)


def component_rank(model_name: str, model, name: str, weight: torch.Tensor, keep_ratio: float) -> int:
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


def choose_method(weight: torch.Tensor, mode: str) -> str:
    rows, columns = weight.shape
    if mode == "osop":
        return "osop"
    if mode == "whitening":
        return "whitening"
    if mode == "hybrid":
        return "osop" if rows <= columns else "whitening"
    raise ValueError(f"Unknown compression mode: {mode}")


def build_svd_modules(model_name: str, model, layer, keep_ratio: float):
    if "llama" in model_name or "vicuna" in model_name:
        return (
            SVD_LlamaAttention(config=model.config, ratio=keep_ratio),
            SVD_LlamaMLP(
                hidden_size=layer.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=keep_ratio,
            ),
            None,
        )
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
            attention_masks.append(kwargs["attention_mask"].detach().cpu())
            if "opt" not in model_name:
                position_ids.append(kwargs["position_ids"].detach().cpu())
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
    attention_masks = torch.cat(attention_masks, dim=0)
    if "opt" in model_name:
        position_ids = None
    else:
        position_ids = torch.cat(position_ids, dim=0)
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
):
    print("Collecting calibration activations for OSOP...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_transformer_layers(model_name, model)
    inps, attention_masks, position_ids = collect_first_layer_inputs(model_name, model, calib_loader, dev)
    dtype = next(iter(model.parameters())).dtype

    print("Start OSOP compression...")
    for layer_idx in tqdm(range(len(layers))):
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        accumulators: Dict[str, OSOPAccumulator] = {}
        method_counts = {"osop": 0, "whitening": 0}

        for name, module in subset.items():
            method = choose_method(module.weight, mode)
            method_counts[method] += 1
            rank = component_rank(model_name, model, name, module.weight, keep_ratio)
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
            attn = attention_masks[sample_idx:sample_idx + 1].to(dev)
            if "opt" in model_name:
                out = layer(inp, attention_mask=attn)[0]
            else:
                pos = position_ids[sample_idx:sample_idx + 1].to(dev)
                out = layer(inp, attention_mask=attn, position_ids=pos)[0]
            outs.append(out.detach().cpu())

        for handle in handles:
            handle.remove()

        svd_attn, svd_mlp, svd_decoder = build_svd_modules(model_name, model, layer, keep_ratio)
        for name, accumulator in accumulators.items():
            a, b = accumulator.factors()
            assign_factor(model_name, layer, svd_attn, svd_mlp, svd_decoder, name, a, b, dtype)

        if "opt" in model_name:
            svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
            svd_decoder.final_layer_norm = layer.final_layer_norm
            layers[layer_idx] = svd_decoder.cpu()
        else:
            layers[layer_idx] = layer.cpu()

        inps = torch.cat(outs, dim=0)
        del outs, accumulators, layer
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
    parser.add_argument("--gamma_mode", type=str, default="ones", choices=["ones", "row_norm", "row_abs_mean"])
    parser.add_argument("--gamma_path", type=str, default=None, help="Optional torch file: {layer_idx: {linear_name: gamma_diag}}.")
    parser.add_argument("--damping", type=float, default=1e-6)
    parser.add_argument("--accum_dtype", type=str, default="float32", choices=["float32", "float64"])
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
        )
        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
            save_name = (
                args.model.replace("/", "_").replace("-", "_")
                + f"_osop_{args.mode}_{keep_ratio}.pt"
            )
            torch.save({"model": model, "tokenizer": tokenizer}, os.path.join(args.save_path, save_name))
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
        model.eval()
        model = model.float().to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=["wikitext2"], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
