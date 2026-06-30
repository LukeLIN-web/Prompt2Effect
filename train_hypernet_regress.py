r"""
Stage-1: train the Prompt2Effect HyperNet to regress onto a bank of pre-trained per-effect
teacher LoRAs (NMSE / joint factor loss). See scripts/train_hypernet_regress.sh for a launcher.

Example:
  torchrun --nproc_per_node=8 -m train_hypernet_regress --distributed \
    --base-ckpt /path/to/Wan2.1-Original \
    --lora-ckpt /path/to/wan_lora/squish.safetensors  [--lora-ckpt ... for each teacher LoRA] \
    --text-embed-bank ./results/wan_effects_text_bank.pt \
    --dataset-cache ./results/effects_wan.pt \
    --hyper-version ve --hidden-dim 2048 --n-layer 14 --n-heads 32 --include_emb \
    --loss nmse --lr 5e-5 --precision bf16 --max-grad-norm 1.0 \
    --weight-regex "(self_attn|cross_attn|ffn)\.(q|k|v|k_img|v_img|o|0|2)" \
    --save ./results/hypernet_regress/wan --save_step 500
"""
import os
import re
import math
import json
import argparse
import tempfile
import gc
import glob
import datetime
from typing import Callable, Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from collections.abc import Mapping
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from safetensors.torch import load_file as safe_load_file
from safetensors import safe_open

# Import local hypernet components
from hypernet import HyperConfig, HyperNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =====================================================
# Dataclasses and dataset containers
# =====================================================

@dataclass
class LoRAPair:
    A: torch.Tensor  # [D, r]
    B: torch.Tensor  # [r, D]


@dataclass
class ModuleExample:
    name: str
    layer_index: int
    base_weight: torch.Tensor  # [D, D]
    A: torch.Tensor            # [D, r]
    B: torch.Tensor            # [r, D]
    module_type: str
    lora_id: int               # which LoRA this sample comes from (0..N-1)
    # Cached base weight SVD subspaces for rotation regularization
    U_r: torch.Tensor   # [D, r]  (left singular vectors)
    V_r: torch.Tensor   # [D, r]  (right singular vectors, i.e., V)


class ModuleDataset:
    def __init__(self, items: List[ModuleExample]):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int) -> ModuleExample:
        return self.items[idx]


# =========================
# Dataset save/load (cache)
# =========================
def _to_cpu_fp32(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().to("cpu", dtype=torch.float32).contiguous()


def _module_example_to_record(ex: ModuleExample) -> dict:
    # base_weight is deduped separately; here we store a reference by name
    return {
        "name": ex.name,
        "layer_index": int(ex.layer_index),
        "A": _to_cpu_fp32(ex.A),
        "B": _to_cpu_fp32(ex.B),
        "module_type": ex.module_type,
        "lora_id": int(ex.lora_id),
        "U_r": _to_cpu_fp32(ex.U_r),
        "V_r": _to_cpu_fp32(ex.V_r),
    }


def save_dataset_to_file(dataset: ModuleDataset, path: str, D: int, r: int, num_loras: int, meta: Optional[dict] = None,):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Deduplicate base weights keyed by module name
    base_weights: Dict[str, torch.Tensor] = {}
    for ex in dataset.items:
        if ex.name not in base_weights:
            base_weights[ex.name] = _to_cpu_fp32(ex.base_weight)

    records = [_module_example_to_record(ex) for ex in dataset.items]
    payload = {
        "format_version": 2,
        "D": int(D),
        "r": int(r),
        "num_loras": int(num_loras),
        "meta": meta or {},
        "base_weights": base_weights,  # { name -> [D,D] tensor }
        "items": records,              # list of dicts (reference base by name)
    }
    torch.save(payload, path)
    print(f"[dataset] saved {len(records)} items to {path} (D={D}, r={r}, num_loras={num_loras})")


def _record_to_module_example(rec: dict, base_weights: Dict[str, torch.Tensor]) -> ModuleExample:
    name = rec["name"]
    return ModuleExample(
        name=name,
        layer_index=int(rec["layer_index"]),
        base_weight=base_weights[name],  # [D, D]
        A=rec["A"].contiguous(),
        B=rec["B"].contiguous(),
        module_type=rec["module_type"],
        lora_id=int(rec["lora_id"]),
        U_r=rec.get("U_r"),
        V_r=rec.get("V_r"),
    )


def crop_dataset_rank_inplace(dataset: ModuleDataset, r_keep: int):
    """In-place: keep only the top-k (already SVD-ordered) channels for every example."""
    if r_keep is None:
        return
    for ex in dataset.items:
        r_here = int(ex.A.shape[1])
        if r_keep > r_here:
            raise ValueError(f"[target-topr] Requested {r_keep} but example '{ex.name}' has rank {r_here}.")
        if r_keep == r_here:
            continue
        ex.A   = ex.A[:, :r_keep].contiguous()
        ex.B   = ex.B[:r_keep, :].contiguous()
        if ex.U_r is not None:
            ex.U_r = ex.U_r[:, :r_keep].contiguous()
        if ex.V_r is not None:
            ex.V_r = ex.V_r[:, :r_keep].contiguous()


def load_dataset_from_file(path: str) -> Tuple[ModuleDataset, int, int, int, dict]:
    # Use weights_only=False to avoid warning for our own trusted dataset files
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fmt = int(payload.get("format_version", 1))
    if fmt not in (1, 2):
        raise RuntimeError(f"Unknown dataset format version: {fmt}")

    D = int(payload["D"])
    r = int(payload["r"])
    num_loras = int(payload.get("num_loras", 1))
    meta = dict(payload.get("meta", {}))
    base_weights: Dict[str, torch.Tensor] = payload["base_weights"]
    recs = payload["items"]

    items = [_record_to_module_example(rec, base_weights) for rec in recs]
    ds = ModuleDataset(items)
    print(f"[dataset] loaded {len(ds)} items from {path} (D={D}, r={r}, num_loras={num_loras})")
    return ds, D, r, num_loras, meta


# -------------------------
# Gauge-fix utilities for factor supervision
# -------------------------
def _ddp_mean_inplace(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= float(world_size)
    return x


def _svd_canonicalize_pair(A: torch.Tensor, B: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        delta = A @ B  # [D,D]
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)  # W = U diag(S) V^T

        U_r  = U[:, :r].contiguous()
        S_r  = S[:r].contiguous()
        Vh_r = Vh[:r, :].contiguous()

        # Sign-fix: flip each singular vector so Vh_r[row]'s largest-magnitude entry is positive.
        kmax  = torch.argmax(Vh_r.abs(), dim=1)                        # [r]
        signs = torch.sign(Vh_r[torch.arange(r), kmax])
        signs[signs == 0] = 1.0
        U_r  = U_r * signs.unsqueeze(0)                                 # column-wise flip
        Vh_r = Vh_r * signs.unsqueeze(1)                                # row-wise flip

        S_sqrt = S_r.clamp_min(0).sqrt()
        A_can  = U_r * S_sqrt.unsqueeze(0)                              # [D,r]
        B_can  = S_sqrt.unsqueeze(1) * Vh_r                              # [r,D]
    return A_can.contiguous(), B_can.contiguous()


def _dtype_for_precision(precision: str):
    p = precision.lower()
    if p == "bf16": return torch.bfloat16
    if p == "fp16": return torch.float16
    return torch.float32


def extract_module_type(param_name: str, weight_regex: Optional[str]) -> str:
    name = _normalize_prefixes(strip_weight_suffix(param_name))
    grp_pat = re.compile(weight_regex)
    m = grp_pat.search(name)
    return f"{m.group(1)}.{m.group(2)}"


def _parse_csv_list(str_list: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if not str_list:
        return out
    for s in str_list:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out.extend(parts)
    return out


def _parse_csv_int_list(str_list: Optional[List[str]]) -> List[int]:
    raw = _parse_csv_list(str_list)
    ints: List[int] = []
    for p in raw:
        try:
            ints.append(int(p))
        except ValueError:
            raise ValueError(f"Expected integer in --only-layer-indices, got '{p}'")
    return ints


def _pick_subset_all_loras(
    dataset: ModuleDataset,
    num_loras: int,
    only_module: Optional[str],
    only_module_types: Optional[List[str]],
    only_layer_indices: Optional[List[int]],
    world_size: int,
    batch_size: int,
    ddp_drop_last: bool,
):
    rx = re.compile(only_module) if only_module else None
    mt_set = set(only_module_types) if only_module_types else None
    li_set = set(only_layer_indices) if only_layer_indices else None

    # Map (module_type, layer_index) -> {lora_id: ModuleExample}
    by_key: Dict[Tuple[str, int], Dict[int, ModuleExample]] = {}
    for ex in dataset.items:
        if mt_set and ex.module_type not in mt_set:
            continue
        if li_set is not None and ex.layer_index not in li_set:
            continue
        if rx and not rx.search(ex.name):
            continue
        key = (ex.module_type, ex.layer_index)
        m = by_key.setdefault(key, {})
        m[ex.lora_id] = ex

    # Keep only keys that exist in ALL LoRAs
    chosen_keys = [k for k, m in by_key.items() if len(m) == num_loras]
    if not chosen_keys:
        example_keys = sorted({(ex.module_type, ex.layer_index) for ex in dataset.items})[:30]
        raise RuntimeError(
            "[overfit-one] No (module_type, layer_index) satisfied the multi-select filters across all LoRAs. "
            f"Filters: module_types={sorted(mt_set) if mt_set else None}, layers={sorted(li_set) if li_set else None}, regex={only_module!r}. "
            f"First 30 available keys: {example_keys}"
        )
    # Stable order: by (layer_index then module_type)
    chosen_keys = sorted(chosen_keys, key=lambda k: (k[1], k[0]))

    # Rebuild a compact dataset with only the chosen keys, across all LoRAs
    new_items: List[ModuleExample] = []
    for (mt, li) in chosen_keys:
        m = by_key[(mt, li)]
        # Keep LoRA id order stable
        for lid in range(num_loras):
            ex = m[lid]
            new_items.append(ex)
    ds = ModuleDataset(new_items)
    return ds, chosen_keys


# -------------------------
# Utilities for checkpoint I/O
# -------------------------
def _flatten_tensor_mapping(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    flat: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(obj):
        if prefix:
            flat[prefix] = obj
        return flat

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            name = f"{prefix}.{k}" if prefix else k
            flat.update(_flatten_tensor_mapping(v, name))
        return flat

    if hasattr(obj, "state_dict"):
        try:
            sd = obj.state_dict()
            return _flatten_tensor_mapping(sd, prefix)
        except Exception:
            return {}
    return {}


def _maybe_get_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        if all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
            return {k: v for k, v in obj.items() if torch.is_tensor(v)}
        for key in ["state_dict", "model", "module", "model_state_dict", "weights", "state"]:
            if key in obj and isinstance(obj[key], Mapping):
                inner = obj[key]
                flat = _flatten_tensor_mapping(inner)
                if flat:
                    return flat
        flat = _flatten_tensor_mapping(obj)
        if flat:
            return flat
    if hasattr(obj, "state_dict"):
        try:
            sd = obj.state_dict()
            return _maybe_get_state_dict(sd)
        except Exception:
            pass
    raise ValueError("Could not interpret the checkpoint as a state_dict-like object.")


def parse_epoch_step_from_ckpt(path: str) -> tuple[Optional[int], Optional[int]]:
    """Parse E,S from .../hypernet_epoch_<E>_step_<S>.pt (S optional)."""
    base = os.path.basename(path)
    m = re.search(r"hypernet(?:_epoch)?_(\d+)(?:_step_(\d+))?\.pt$", base)
    if not m:
        return None, None
    e = int(m.group(1))
    s = int(m.group(2)) if m.group(2) is not None else 0
    return e, s


def split_root_name_from_save(save_dir: Optional[str], fallback_name: Optional[str] = None
                              ) -> tuple[str, str, str]:
    if save_dir:
        exp_dir = os.path.abspath(save_dir)
        root = os.path.dirname(exp_dir)
        name = os.path.basename(exp_dir)
    else:
        root = os.path.abspath("./results")
        name = (fallback_name or "run")
        exp_dir = os.path.join(root, name)
    return root, name, exp_dir


def load_hypernet_checkpoint(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    extra = {}
    if isinstance(obj, dict) and "state_dict" in obj:
        version = obj.get("hyper_version", None)
        cfg = obj.get("config", None)
        sd = obj["state_dict"]
        if not isinstance(sd, dict):
            sd = _maybe_get_state_dict(sd)
        # Collect optional training state for resume.
        for k in ("optimizer", "scheduler", "epoch", "global_step", "scheduler_meta"):
            if k in obj:
                extra[k] = obj[k]
        return version, cfg, sd, extra
    # Fallback: treat the entire object as a state_dict-like tree
    sd = _maybe_get_state_dict(obj)
    return None, None, sd, extra


def _is_dcp_path(path: str) -> bool:
    if path.endswith(".distcp"):
        return True
    if os.path.isdir(path):
        try:
            for n in os.listdir(path):
                if n.endswith(".distcp") or n == ".metadata" or n.endswith(".metadata"):
                    return True
        except Exception:
            pass
    return False


def _load_dcp_to_state_dict(dcp_root: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    tmp_dir = tempfile.mkdtemp(prefix="dcp_convert_")
    out_file = os.path.join(tmp_dir, "converted.pt")
    print(f"[load_state] Converting DCP at '{dcp_root}' to torch.save -> {out_file}")
    dcp_to_torch_save(dcp_root, out_file)
    obj = torch.load(out_file, map_location=map_location, weights_only=False)
    return _maybe_get_state_dict(obj)


def load_state(path: str, map_location: str = "cpu", target_layers: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    def _keep(k: str) -> bool:
        if target_layers is None:
            return True
        return first_int_in_string(k) in target_layers

    # Single safetensors file (lazy loading).
    if path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(path, framework="pt", device=map_location) as f:
            for k in f.keys():
                if _keep(k):
                    state_dict[k] = f.get_tensor(k)
        return _maybe_get_state_dict(state_dict)

    # Diffusers directory structure (lazy loading).
    if os.path.isdir(path) and not _is_dcp_path(path):
        # Look in transformer subfolder first, then root
        st_files = glob.glob(os.path.join(path, "transformer", "*.safetensors"))
        if not st_files:
            st_files = glob.glob(os.path.join(path, "*.safetensors"))

        if st_files:
            print(f"[load_state] Loading {len(st_files)} safetensors from {path} (Filtered: {target_layers is not None})")
            state_dict = {}
            for f_path in st_files:
                with safe_open(f_path, framework="pt", device=map_location) as f:
                    for k in f.keys():
                        if _keep(k):
                            state_dict[k] = f.get_tensor(k)
            return _maybe_get_state_dict(state_dict)

    # PyTorch (.pt) / DCP checkpoints.
    if _is_dcp_path(path):
        if path.endswith(".distcp"):
            dcp_root = os.path.dirname(path)
        else:
            dcp_root = path
        print(f"[load_state] Detected DCP checkpoint at: {dcp_root} (converting to torch.save format...)")
        sd = _load_dcp_to_state_dict(dcp_root, map_location=map_location)
    else:
        obj = torch.load(path, map_location=map_location, weights_only=False)
        sd = _maybe_get_state_dict(obj)

    # Filter standard .pt files to drop RAM usage quickly
    if target_layers is not None:
        sd = {k: v for k, v in sd.items() if _keep(k)}

    return sd


def first_int_in_string(s: str) -> int:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def strip_weight_suffix(name: str) -> str:
    return name[:-7] if name.endswith(".weight") else name


def discover_square_weights(
    state: Dict[str, torch.Tensor],
    weight_regex: Optional[str] = None,
    force_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    pat = re.compile(weight_regex) if weight_regex else None
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not k.endswith(".weight"):
            continue
        if v is None or not torch.is_tensor(v):
            continue
        if v.ndim != 2:
            continue
        D2, D = v.shape  # [out, in]
        if D2 != D:
            continue
        if pat and not pat.search(k):
            continue
        if force_dim is not None and D != force_dim:
            continue
        out[strip_weight_suffix(k)] = v.detach().to(torch.float32).cpu()
    return out


def _normalize_lora_shapes(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"LoRA A/B must be 2D. Got A={A.shape}, B={B.shape}.")
    def _is_Dr(mat):  # [D,r]
        return mat.shape[0] >= mat.shape[1]
    def _is_rD(mat):  # [r,D]
        return mat.shape[0] < mat.shape[1]
    A_norm = A if _is_Dr(A) else A.t()
    B_norm = B if _is_rD(B) else B.t()
    if A_norm.shape[0] != B_norm.shape[1] or A_norm.shape[1] != B_norm.shape[0]:
        A_norm = A if A.shape[0] > A.shape[1] else A.t()
        B_norm = B if B.shape[0] < B.shape[1] else B.t()
    return A_norm.contiguous().float(), B_norm.contiguous().float()


def _normalize_prefixes(name: str) -> str:
    name = name.replace("._fsdp_wrapped_module.", ".")
    name = name.replace("._checkpoint_wrapped_module.", ".")
    drop_anywhere = {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"}
    parts = [p for p in name.split(".") if p not in drop_anywhere]
    name = ".".join(parts)

    for pref in ("submodule.", "module.", "model.", "models.", "backbone.", "net.", "transformer.", "base_model.model.", "diffusion_model."):
        if name.startswith(pref):
            name = name[len(pref):]
    return name


def discover_lora_pairs(lora_state: Dict[str, torch.Tensor], include_regex: Optional[str] = None,) -> Dict[str, LoRAPair]:
    pat = re.compile(include_regex) if include_regex else None
    groups: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in lora_state.items():
        if not torch.is_tensor(v):
            continue
        if pat and not pat.search(k):
            continue

        name = k[:-7] if k.endswith(".weight") else k

        if name.endswith(".lora_A") or name.endswith(".lora.down"):
            base = name.replace(".lora_A", "").replace(".lora.down", "")
            groups.setdefault(base, {})["A"] = v
        elif name.endswith(".lora_B") or name.endswith(".lora.up"):
            base = name.replace(".lora_B", "").replace(".lora.up", "")
            groups.setdefault(base, {})["B"] = v

    result: Dict[str, LoRAPair] = {}
    for base, parts in groups.items():
        if "A" not in parts or "B" not in parts:
            continue
        A, B = _normalize_lora_shapes(parts["A"], parts["B"])
        new_base = _normalize_prefixes(base)
        result[new_base] = LoRAPair(A=A, B=B)
    return result


def build_dataset_multi(
    base_ckpt: str,
    lora_ckpts: List[str],
    weight_regex: str,
    device: torch.device,
    force_dim: Optional[int] = None,
    canon_teacher: bool = True,
    target_layers: Optional[List[int]] = None,
) -> Tuple[ModuleDataset, int, int, int]:
    base_state = load_state(base_ckpt, map_location="cpu", target_layers=target_layers)
    base_map = discover_square_weights(base_state, weight_regex=weight_regex, force_dim=force_dim)

    items: List[ModuleExample] = []
    D_inferred: Optional[int] = None
    r_inferred: Optional[int] = None
    # Cache base SVD subspaces once per module name (reused across LoRAs).
    basis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for lid, lora_ckpt in enumerate(lora_ckpts):
        lora_state = load_state(lora_ckpt, map_location="cpu", target_layers=target_layers)
        lora_map = discover_lora_pairs(lora_state, include_regex=weight_regex)
        matched = 0
        for base_name, W in base_map.items():
            if base_name not in lora_map:
                continue
            pair = lora_map[base_name]
            D = W.shape[0]
            r = pair.B.shape[0]
            if D_inferred is None:
                D_inferred = D
            if r_inferred is None:
                r_inferred = r
            if D != D_inferred or r != r_inferred:
                continue

            # (1) Canonicalize the ground-truth LoRA weights via SVD.
            A_tgt, B_tgt = pair.A.contiguous().float(), pair.B.contiguous().float()
            if canon_teacher:
                A_tgt, B_tgt = _svd_canonicalize_pair(A_tgt, B_tgt, r_inferred)

            # (2) Cache base SVD subspaces (used for rotation-regularization metadata).
            if base_name not in basis_cache:
                U, _, Vh = torch.linalg.svd(W, full_matrices=False)  # W: [D, D]
                U_r = U[:, :r_inferred].contiguous()
                V_r = Vh[:r_inferred, :].transpose(0, 1).contiguous()  # [D, r]
                basis_cache[base_name] = (U_r.cpu(), V_r.cpu())
            U_r, V_r = basis_cache[base_name]

            mod_type = extract_module_type(base_name, weight_regex)
            items.append(ModuleExample(
                name=base_name,
                layer_index=first_int_in_string(base_name),
                base_weight=W.contiguous().float(),
                A=A_tgt.cpu(),  # store canonicalized
                B=B_tgt.cpu(),
                module_type=mod_type,
                lora_id=lid,
                U_r=U_r,
                V_r=V_r
            ))
            matched += 1

        if is_main_process():
            print(f"[dataset] LoRA[{lid}] '{lora_ckpt}': matched {matched} modules (canon={canon_teacher}).")

    if not items:
        raise RuntimeError("No overlapping modules between base and the provided LoRAs after filtering. "
                           "Check --weight-regex and the checkpoint contents.")

    return ModuleDataset(items), int(D_inferred), int(r_inferred), len(lora_ckpts)


def build_hypernet(version: str, text_dim: int, weight_dim: int, rank: int,
                   hidden_dim: int, n_heads: int, n_layer: int, dropout: float,
                   include_emb: bool, module_specs: Optional[Dict[str,dict]] = None,
                   max_layers: Optional[int] = None, num_latents: int = 256):
    version = version.lower()
    cfg = HyperConfig(
        text_dim=text_dim,
        weight_dim=weight_dim,
        module_specs=module_specs,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layer=n_layer,
        dropout=dropout,
        rank=rank,
        alpha_scale=1.0,
        include_emb=include_emb,
        max_layers=(max_layers if max_layers is not None else 32),
        layer_embed_dim=32,
        module_embed_dim=32,
        use_sin_pos=True,
        grad_ckpt=False,
    )
    if version != "ve":
        raise ValueError(f"Unknown HyperNet version: {version!r} (only 've' is supported).")
    cfg.num_latents = num_latents
    return HyperNet(cfg)


def nmse_on_AB(pred, target_A, target_B, eps: float = 1e-8) -> torch.Tensor:
    """Relative (power-normalized) MSE over A and B, averaged over the batch."""
    A_pred = pred.get("Lora_A", pred.get("A"))
    B_pred = pred.get("Lora_B", pred.get("B"))
    if A_pred is None or B_pred is None:
        raise KeyError("HyperNet forward() must return 'Lora_A'/'Lora_B' (or 'A'/'B').")

    # Per-sample Frobenius errors and energies
    errA = (A_pred - target_A).pow(2).sum(dim=(1, 2))
    powA = target_A.pow(2).sum(dim=(1, 2))
    errB = (B_pred - target_B).pow(2).sum(dim=(1, 2))
    powB = target_B.pow(2).sum(dim=(1, 2))

    nmseA = errA / (powA + eps)
    nmseB = errB / (powB + eps)
    return 0.5 * (nmseA.mean() + nmseB.mean())


def nmse_on_deltaW(pred, target_A, target_B, eps: float = 1e-8, uv: bool = False) -> torch.Tensor:
    if "deltaW" in pred:
        dW_pred = pred["deltaW"]
    elif uv:
        sigma = torch.diag_embed(pred["sigma"]).to(pred["U"].dtype)
        dW_pred = torch.bmm(torch.bmm(pred["U"], sigma), pred["V"])
    else:
        A_pred = pred.get("Lora_A", pred.get("A"))
        B_pred = pred.get("Lora_B", pred.get("B"))
        dW_pred = torch.matmul(A_pred, B_pred)     # [B, D, D]
    dW_tgt  = torch.matmul(target_A, target_B) # [B, D, D]
    err = (dW_pred - dW_tgt).pow(2).sum(dim=(1, 2))
    pow = dW_tgt.pow(2).sum(dim=(1, 2)) + eps
    return (err / pow).mean()


def _crop_AB_batched(A: torch.Tensor, B: torch.Tensor, r_keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    If r_keep < current rank: crop.
    If r_keep > current rank: pad with zeros along the rank dimension.
    A: [B, D, r], B: [B, r, D]
    """
    r_now = A.shape[-1]
    if r_keep == r_now:
        return A, B
    if r_keep < r_now:
        return A[..., :r_keep].contiguous(), B[:, :r_keep, :].contiguous()
    # pad
    pad_a = torch.zeros(A.shape[0], A.shape[1], r_keep - r_now, device=A.device, dtype=A.dtype)
    pad_b = torch.zeros(B.shape[0], r_keep - r_now, B.shape[2], device=B.device, dtype=B.dtype)
    A2 = torch.cat([A, pad_a], dim=-1)
    B2 = torch.cat([B, pad_b], dim=1)
    return A2.contiguous(), B2.contiguous()


def match_ab_ranks_for_loss(
    A_pred: torch.Tensor, B_pred: torch.Tensor,
    A_tgt: torch.Tensor,  B_tgt: torch.Tensor,
    mode: str = "min"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (A_pred_m, B_pred_m, A_tgt_m, B_tgt_m) with ranks matched per policy."""
    r_pred = int(A_pred.shape[-1])
    r_tgt  = int(A_tgt.shape[-1])
    if r_pred == r_tgt:
        return A_pred, B_pred, A_tgt, B_tgt
    if mode == "min":
        k = min(r_pred, r_tgt)
        return A_pred[..., :k], B_pred[:, :k, :], A_tgt[..., :k], B_tgt[:, :k, :]
    if mode == "pred":
        return A_pred, B_pred, A_tgt[..., :r_pred], B_tgt[:, :r_pred, :]
    if mode == "target":
        A_pred_m, B_pred_m = _crop_AB_batched(A_pred, B_pred, r_tgt)
        return A_pred_m, B_pred_m, A_tgt, B_tgt
    if mode == "pad":
        k = max(r_pred, r_tgt)
        A_pred_m, B_pred_m = _crop_AB_batched(A_pred, B_pred, k)
        A_tgt_m,  B_tgt_m  = _crop_AB_batched(A_tgt,  B_tgt,  k)
        return A_pred_m, B_pred_m, A_tgt_m, B_tgt_m
    raise ValueError(f"Unknown --ab-rank-mode: {mode}")


def compute_regression_loss(
    out: Dict[str, Any],
    A_tgt: torch.Tensor,
    B_tgt: torch.Tensor,
    loss_type: str,
    ab_rank_mode: str,
    nmse_eps: float,
    nmse_delta_w: float,
) -> torch.Tensor:
    """Regression loss between predicted and teacher LoRA factors.

    Supports NMSE on factors (``nmse``), NMSE on the composed update ``ΔW``
    (``nmse_delta``), or a weighted mix of the two (``joint``).
    """
    A_pred = out["Lora_A"]
    B_pred = out["Lora_B"]
    if loss_type == "nmse":
        A_p, B_p, A_t, B_t = match_ab_ranks_for_loss(A_pred, B_pred, A_tgt, B_tgt, mode=ab_rank_mode)
        return nmse_on_AB({"Lora_A": A_p, "Lora_B": B_p}, A_t, B_t, eps=nmse_eps)
    if loss_type == "nmse_delta":
        return nmse_on_deltaW(out, A_tgt, B_tgt, eps=nmse_eps)
    if loss_type == "joint":
        A_p, B_p, A_t, B_t = match_ab_ranks_for_loss(A_pred, B_pred, A_tgt, B_tgt, mode=ab_rank_mode)
        loss_ab = nmse_on_AB({"Lora_A": A_p, "Lora_B": B_p}, A_t, B_t, eps=nmse_eps)
        loss_dw = nmse_on_deltaW(out, A_tgt, B_tgt, eps=nmse_eps)
        return (1 - nmse_delta_w) * loss_ab + nmse_delta_w * loss_dw
    raise ValueError(f"Unknown --loss {loss_type}")


# -------------------------
# DDP / AMP helpers
# -------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def setup_distributed(enabled: bool, device_arg: str):
    # Check if we are being launched by torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        enabled = True
    if not enabled:
        device = torch.device(device_arg)
        return device, 0, 1, False
    # Initialize DDP
    dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=7200)
        )
    # Get rank info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set the device to the specific GPU for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"[DDP] Initialized: rank={rank}, world_size={world_size}, backend={dist.get_backend()}")
    return device, rank, world_size, True

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def collate_examples(batch: List[ModuleExample]):
    unique_weights = {}
    indices = []
    for i, ex in enumerate(batch):
        if ex.name not in unique_weights:
            unique_weights[ex.name] = (len(unique_weights), ex.base_weight)
        indices.append(unique_weights[ex.name][0])

    # Sort unique weights by their discovered index to ensure consistent order
    sorted_unique = sorted(unique_weights.values(), key=lambda x: x[0])
    unique_base_weights = torch.stack([x[1] for x in sorted_unique], dim=0) # [N_unique, D, D]
    reconstruct_idx = torch.tensor(indices, dtype=torch.long) # [B]

    # Standard collation for others
    A_tgt = torch.stack([ex.A for ex in batch], dim=0)
    B_tgt = torch.stack([ex.B for ex in batch], dim=0)
    U_r   = torch.stack([ex.U_r for ex in batch], dim=0) if batch[0].U_r is not None else None
    V_r   = torch.stack([ex.V_r for ex in batch], dim=0) if batch[0].V_r is not None else None
    module_types = [ex.module_type for ex in batch]
    layer_idx = torch.tensor([ex.layer_index for ex in batch], dtype=torch.long)
    lora_ids = torch.tensor([ex.lora_id for ex in batch], dtype=torch.long)

    return unique_base_weights, reconstruct_idx, A_tgt, B_tgt, module_types, layer_idx, lora_ids, U_r, V_r


def build_constant_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    # Keep behavior robust for tiny runs
    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
    if total_steps <= 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    def lr_lambda(current_step: int) -> float:
        return float(current_step + 1) / float(warmup_steps) if (warmup_steps > 0 and current_step < warmup_steps) else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



def build_cosine_warmup_scheduler(optimizer: torch.optim.Optimizer, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int):
    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
    if total_steps <= 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    min_factor = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps + 1:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# -------------------------
# Training loop (logs per step)
# -------------------------
def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    text_embeds_by_lora: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    world_size: int,
    log_every: int,
    log_epoch: int,
    precision: str,
    use_wandb: bool,
    global_step_start: int,
    loss_type: str,
    nmse_eps: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    nmse_delta_w: float = 0.0,
    max_grad_norm: float = None,
    grad_norm_type: float = 2.0,
    num_loras: int = 1,
    ab_rank_mode: str = "min",
    masks_by_lora: Optional[List[torch.Tensor]] = None,
    text_bank_multi: Optional[torch.Tensor] = None,
    mask_bank_multi: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[float, int]:
    net.train()
    step_in_epoch = 0
    epoch_loss_sum_local = 0.0

    # Select the autocast context. Fall back to fp32 (nullcontext) when bf16 is
    # requested but unsupported on the current CUDA device.
    use_autocast = precision == "bf16" and (
        (device.type == "cuda" and torch.cuda.is_bf16_supported()) or device.type == "cpu"
    )
    if use_autocast and device.type == "cuda":
        compute_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        compute_ctx = nullcontext()
    target_dtype = _dtype_for_precision(precision)
    bank_legacy = None
    mask_legacy = None
    if (text_bank_multi is None) or (offsets is None):
        assert text_embeds_by_lora is not None and len(text_embeds_by_lora) == num_loras
        bank_legacy = torch.stack(text_embeds_by_lora, dim=0).to(device=device, dtype=target_dtype)
        if masks_by_lora is not None:
            assert len(masks_by_lora) == num_loras
            mask_legacy = torch.stack(masks_by_lora, dim=0).to(device=device, dtype=torch.bool)

    for batch in loader:
        if len(batch) == 9:
            unique_base, base_idx, A_tgt, B_tgt, module_type, layer_idx, lora_ids, U_r, V_r = batch
        elif len(batch) == 7:
            unique_base, base_idx, A_tgt, B_tgt, module_type, layer_idx, lora_ids = batch
            U_r, V_r = None, None
        # Transfer the small set of unique weights, then expand on the GPU (the
        # device-side index_select is far cheaper than transferring the full batch).
        unique_base = unique_base.to(device=device, dtype=target_dtype, non_blocking=True)
        base_idx = base_idx.to(device=device, non_blocking=True)
        base_weight = unique_base.index_select(0, base_idx)
        A_tgt = A_tgt.to(device=device, dtype=target_dtype, non_blocking=True)
        B_tgt = B_tgt.to(device=device, dtype=target_dtype, non_blocking=True)
        if isinstance(layer_idx, torch.Tensor):
            layer_idx = layer_idx.to(device=device, non_blocking=True)
        lora_ids = lora_ids.to(device=device, non_blocking=True)

        if U_r is not None:
            U_r = U_r.to(device=device, dtype=target_dtype, non_blocking=True)
        if V_r is not None:
            V_r = V_r.to(device=device, dtype=target_dtype, non_blocking=True)

        if (text_bank_multi is not None) and (offsets is not None):
            B = lora_ids.shape[0]
            idxs: List[int] = []
            # Sample one prompt embedding per LoRA from its slice of the bank.
            for b in range(B):
                lid = int(lora_ids[b].item())
                s = int(offsets[lid].item())
                e = int(offsets[lid + 1].item())
                if e <= s:
                    raise RuntimeError(f"LoRA {lid} has empty range in offsets")
                idx = int(torch.randint(low=s, high=e, size=(1,)).item())
                idxs.append(idx)
            idxs_t = torch.tensor(idxs, dtype=torch.long)               # CPU
            batch_text = text_bank_multi.index_select(0, idxs_t)        # [B, T, E] on CPU
            text = batch_text.to(device=device, dtype=target_dtype, non_blocking=True)
            if mask_bank_multi is not None:
                # Pre-computed padding mask (True = pad).
                batch_mask_cpu = mask_bank_multi.index_select(0, idxs_t)
                text_mask = batch_mask_cpu.to(device=device, non_blocking=True)
            else:
                # Fallback: treat exact-zero rows as padding.
                text_mask = (text.abs().sum(dim=-1) == 0)
        else:
            idx = lora_ids.long()
            text = bank_legacy.index_select(0, idx)
            text_mask = mask_legacy.index_select(0, idx) if mask_legacy is not None else None

        with compute_ctx:
            out = net(text, base_weight=base_weight, module_type=module_type,
                      layer_index=layer_idx, text_key_padding_mask=text_mask)
            loss = compute_regression_loss(
                out, A_tgt, B_tgt, loss_type, ab_rank_mode, nmse_eps, nmse_delta_w
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if (max_grad_norm is not None) and (max_grad_norm > 0):
            module_for_clip = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
            torch.nn.utils.clip_grad_norm_(module_for_clip.parameters(), max_grad_norm, norm_type=grad_norm_type)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Reduce loss across ranks
        with torch.no_grad():
            loss_detached = loss.detach()
            if is_dist_avail_and_initialized():
                dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
                loss_detached /= world_size
            reduced_loss = loss_detached.item()

        step_in_epoch += 1
        epoch_loss_sum_local += reduced_loss
        global_step = global_step_start + step_in_epoch


        if (step_in_epoch % log_every) == 0 and (epoch % log_epoch) == 0:
            with torch.no_grad():
                nmse_dW  = nmse_on_deltaW(out, A_tgt, B_tgt, eps=nmse_eps)
                A_p, B_p, A_t, B_t = match_ab_ranks_for_loss(out["Lora_A"], out["Lora_B"], A_tgt, B_tgt, mode=ab_rank_mode)
                nmse_raw = nmse_on_AB({"Lora_A": A_p, "Lora_B": B_p}, A_t, B_t, eps=nmse_eps)  # scalar tensor
                # DDP reduce to global mean
                _ddp_mean_inplace(nmse_raw, world_size)
                _ddp_mean_inplace(nmse_dW,  world_size)

            if is_main_process():
                cur_lr = optimizer.param_groups[0]["lr"]
                line = (f"[epoch {epoch} step {global_step}] "
                        f"loss={loss.detach().item():.6f}  lr={cur_lr:.3e}  "
                        f"nmse_raw={nmse_raw.item():.6f}  nmse_delta={nmse_dW.item():.6f}")
                print(line)

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.detach().item(),
                        "train/lr": cur_lr,
                        "train/nmse_raw": nmse_raw.item(),
                        "train/nmse_delta": nmse_dW.item(),
                        "epoch": epoch,
                        "step": global_step,
                    })

    # Global epoch average
    loss_tensor = torch.tensor([epoch_loss_sum_local], device=device)
    steps_tensor = torch.tensor([step_in_epoch], device=device, dtype=torch.float32)
    if is_dist_avail_and_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)
    epoch_avg_loss = (loss_tensor / torch.clamp(steps_tensor, min=1.0)).item()

    return epoch_avg_loss, global_step_start + step_in_epoch


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        "Stage-1 HyperNet regression trainer (NMSE / joint factor loss) with DDP, "
        "W&B step logging, and mixed precision."
    )
    parser.add_argument("--base-ckpt", type=str, required=False, help="Path to the base model checkpoint (.pt/.pth, state_dict-like). Required unless loading a cached dataset.")
    parser.add_argument("--lora-ckpt", dest="lora_ckpts", action="append", required=False, help="Repeat for multiple LoRAs, or pass a comma-separated list. Required unless loading a cached dataset.")
    parser.add_argument("--dataset-cache", type=str, default=None, help="Convenience: if file exists, load it; else rank 0 builds/saves, then all ranks load.")
    parser.add_argument("--hypernet-ckpt", type=str, default=None, help="Path to a saved HyperNet (.pt) to warm-start and continue training.")
    parser.add_argument("--hypernet-strict", action="store_true", help="Strict key match when loading --hypernet-ckpt (default: non-strict).")
    parser.add_argument("--ignore-hn-config", action="store_true", help="Ignore architecture params stored inside --hypernet-ckpt; build from CLI flags instead.")

    parser.add_argument("--weight-regex", type=str, default=r"(cross_attn|attn)\.(q_proj|k_proj|k_proj_img|v_proj|v_proj_img|proj)", help="Optional regex to select which base weights to use (match on full param name)")
    parser.add_argument("--weight-dim", type=int, default=None, help="Force a specific D (only keep square weights with this size)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="If set (>0), clip global gradient norm to this value each step (DDP-aware).")
    parser.add_argument("--grad-norm-type", type=float, default=2.0, help="p-norm for gradient clipping (e.g., 2.0 or inf).")
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--hyper-version", type=str, default="ve", choices=["ve"])
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--num-latents", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--include_emb", action="store_true")
    parser.add_argument("--save", type=str, default=None, help="Where to save the trained HyperNet (.pt)")
    parser.add_argument("--save_step", type=int, default=100, help="When to store the ckpt (EPOCH-based).")
    parser.add_argument("--text-embed-bank", type=str, default=None,
        help="Path to a .pt produced by encode_prompts.py with {'embeddings':[L,T,E],'names':[L],'prompts':[L]}. "
            "If set, replaces the dummy per-LoRA embeddings. "
            "Order in effect_prompts.txt MUST match the LoRA order in the cached dataset."
    )

    parser.add_argument("--no-canon-teacher", action="store_true", help="Disable SVD-based canonicalization of teacher (A,B) pairs.")

    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"], help="LR schedule: constant or cosine (with optional warmup).")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Cosine floor (final LR).")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Linear warmup steps. If None, uses warmup-ratio * total_steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.01, help="If --warmup-steps is None, warmup steps = ratio * total_steps.")

    # New flags
    parser.add_argument("--distributed", action="store_true", help="Enable DDP (use with torchrun).")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"], help="Mixed precision mode.")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N optimizer steps.")
    parser.add_argument("--log-epoch", type=int, default=1, help="Log every N epochs")
    parser.add_argument("--ddp-drop-last", action="store_true", help="Drop last incomplete batch in DDP to keep step counts equal across ranks.")

    # Rank control
    parser.add_argument("--rank", type=int, default=None,
                        help="Override model (prediction) rank r. Default: use target/dataset rank.")
    parser.add_argument("--target-topr", type=int, default=None,
                        help="Crop ground-truth A/B (and U_r/V_r) to top-k SVD channels (k <= dataset r).")
    parser.add_argument("--ab-rank-mode", type=str, default="min",
                        choices=["min", "pred", "target", "pad"],
                        help="When r_pred != r_target and using factor losses, how to match ranks.")

    # Overfit-one flags
    parser.add_argument("--only-module-types", dest="only_module_types", action="append", default=None,
                        help="Comma-separated or repeatable list of module types (e.g., 'attn.q_proj,cross_attn.q_proj').")
    parser.add_argument("--only-layer-indices", dest="only_layer_indices", action="append", default=None,
                        help="Comma-separated or repeatable list of layer indices (e.g., '12,16,24').")
    parser.add_argument("--overfit-one", action="store_true",
                        help="Keep exactly one ModuleExample and duplicate it to build the dataset.")
    parser.add_argument("--only-module", type=str, default=None,
                        help="Regex to pick the single module from ModuleExample.name (after prefix normalization).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers.")

    # W&B flags
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Existing W&B run id to continue logging into.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, help="one of: online, offline, disabled")

    parser.add_argument("--loss", type=str, default="nmse", choices=["nmse", "nmse_delta", "joint"],
                        help="Supervision: NMSE on factors (nmse), NMSE on ΔW (nmse_delta), or a weighted mix (joint).")
    parser.add_argument("--nmse-delta-w", type=float, default=0.0, help="Mixing weight for the ΔW term when --loss joint.")
    parser.add_argument("--nmse-eps", type=float, default=1e-8, help="Stability epsilon for NMSE denominators.")

    # Allow precise override of the derived global step when resuming from a legacy checkpoint.
    parser.add_argument("--resume-global-step", type=int, default=None,
                        help="Override derived global step when resuming from legacy epoch-only checkpoints.")

    args = parser.parse_args()
    # DDP setup and device selection
    device, rank, world_size, ddp_enabled = setup_distributed(args.distributed, args.device)

    save_root, save_name, exp_dir = split_root_name_from_save(args.save, args.wandb_run_name)
    if is_main_process():
        os.makedirs(exp_dir, exist_ok=True)

    def _need_build_from_ckpts() -> bool:
        if args.dataset_cache and os.path.exists(args.dataset_cache):
            return False
        return True

    # Validate required inputs only if we need to build
    if _need_build_from_ckpts():
        if not args.base_ckpt:
            raise ValueError("Please provide --base-ckpt (or use --dataset-load/--dataset-cache).")
        lora_ckpts: List[str] = []
        for entry in args.lora_ckpts:
            lora_ckpts += [p.strip() for p in entry.split(",") if p.strip()]
        if not lora_ckpts:
            raise ValueError("Please provide at least one --lora-ckpt (or use --dataset-load/--dataset-cache).")

    # Build or load dataset
    dataset: ModuleDataset
    D: int
    r: int
    num_loras: int
    meta: dict = {}

    # Parse layer indices before loading files so we can filter while reading.
    target_layers: Optional[List[int]] = None
    if args.only_layer_indices:
        target_layers = _parse_csv_int_list(args.only_layer_indices)

    if args.dataset_cache:
        cache_path = args.dataset_cache
        if os.path.exists(cache_path):
            dataset, D, r, num_loras, meta = load_dataset_from_file(cache_path)
        else:
            if is_main_process():
                ds, D_b, r_b, n_b = build_dataset_multi(
                    base_ckpt=args.base_ckpt,
                    lora_ckpts=lora_ckpts,
                    weight_regex=args.weight_regex,
                    device=device,
                    force_dim=args.weight_dim,
                    canon_teacher=not args.no_canon_teacher,
                    target_layers=target_layers,
                )
                meta = {
                    "base_ckpt": args.base_ckpt,
                    "lora_ckpts": lora_ckpts,
                    "weight_regex": args.weight_regex,
                    "force_dim": args.weight_dim,
                    "canon_teacher": not args.no_canon_teacher,
                }
                save_dataset_to_file(ds, cache_path, D_b, r_b, n_b, meta=meta)
            if ddp_enabled:
                dist.barrier()
            dataset, D, r, num_loras, meta = load_dataset_from_file(cache_path)

    if is_main_process():
        print(f"Discovered {len(dataset)} module examples across {num_loras} LoRAs with D={D}, dataset_rank={r}")

    # Build module_specs BEFORE any overfit-one reduction so vocab covers all types
    all_module_types = sorted({ex.module_type for ex in dataset.items})
    max_layer_idx = max(ex.layer_index for ex in dataset.items) if dataset.items else 0
    module_specs = {name: {} for name in all_module_types}

    # Optional: collapse to a subset and duplicate it (supports single or multiple keys)
    if args.overfit_one:
        # Merge single-value flags with new list flags
        mt_list: List[str] = []
        mt_list.extend(_parse_csv_list(args.only_module_types))
        mt_list = [m for m in (t.strip() for t in mt_list) if m]

        li_list: List[int] = []
        li_list.extend(_parse_csv_int_list(args.only_layer_indices))

        dataset, chosen_keys = _pick_subset_all_loras(
            dataset=dataset,
            num_loras=num_loras,
            only_module=args.only_module,
            only_module_types=mt_list if mt_list else None,
            only_layer_indices=target_layers,
            world_size=world_size if ddp_enabled else 1,
            batch_size=args.batch_size,
            ddp_drop_last=args.ddp_drop_last,
        )
        if is_main_process():
            preview = ", ".join([f"{mt}[L{li}]" for (mt, li) in chosen_keys[:8]])
            print(f"[overfit-one] multi-select keys={len(chosen_keys)} across {num_loras} LoRAs; "
                    f"First keys: {preview}{' ...' if len(chosen_keys)>8 else ''}")

    # Optionally crop teacher to top-k and choose model rank
    r_data = int(r)
    r_target = int(args.target_topr) if args.target_topr is not None else r_data
    if r_target > r_data:
        raise ValueError(f"--target-topr ({r_target}) cannot exceed dataset rank ({r_data}).")
    if r_target != r_data:
        if is_main_process():
            print(f"[rank] Cropping teacher A/B (and U_r/V_r) to top-{r_target} (dataset r={r_data}).")
        crop_dataset_rank_inplace(dataset, r_target)
    r_model = int(args.rank) if args.rank is not None else r_target
    if r_model > D:
        raise ValueError(f"--rank ({r_model}) must be <= weight dim D ({D}).")
    if is_main_process():
        print(f"[rank] Using model r_model={r_model}, target r_target={r_target}, dataset r_data={r_data}")

    # optionally load a pretrained HyperNet
    ckpt_version = None
    ckpt_cfg = None
    ckpt_state = None
    ckpt_extra = {}
    resume_epoch_from_name = 0
    resume_global_step_from_name = 0
    is_resume = False

    if args.hypernet_ckpt:
        ckpt_version, ckpt_cfg, ckpt_state, ckpt_extra = load_hypernet_checkpoint(args.hypernet_ckpt)
        if is_main_process():
            print(f"[hypernet-ckpt] loaded '{args.hypernet_ckpt}' "
                  f"(version={ckpt_version}, has_config={ckpt_cfg is not None})")
        # Continue training only if same <name> (parent folder)
        ckpt_name = os.path.basename(os.path.dirname(os.path.abspath(args.hypernet_ckpt)))
        if ckpt_name == save_name:
            is_resume = True
            e_file, s_file = parse_epoch_step_from_ckpt(args.hypernet_ckpt)
            resume_epoch_from_name = int(ckpt_extra.get("epoch", e_file or 0))
            resume_global_step_from_name = int(ckpt_extra.get("global_step", s_file or 0))
            if is_main_process():
                print(f"[resume] same run name detected -> resume from epoch={resume_epoch_from_name}, step={resume_global_step_from_name}")
        else:
            if is_main_process():
                print(f"[warm-start] ckpt is not under name '{save_name}' -> weights will load, but training restarts.")

    # Decide architecture to build
    hyper_version_to_use = (ckpt_version or args.hyper_version)
    text_dim_build = args.text_dim
    hidden_dim_build = args.hidden_dim
    n_heads_build = args.n_heads
    n_layer_build = args.n_layer
    dropout_build = args.dropout
    include_emb_build = args.include_emb
    max_layers_build = (max_layer_idx + 1)

    if ckpt_cfg is not None and not args.ignore_hn_config:
        # Basic compatibility check with dataset/base
        if "weight_dim" in ckpt_cfg and int(ckpt_cfg["weight_dim"]) != int(D):
            raise ValueError(f"[hypernet-ckpt] weight_dim mismatch: ckpt={ckpt_cfg['weight_dim']} vs dataset/base={D}")
        text_dim_build     = int(ckpt_cfg.get("text_dim", text_dim_build))
        hidden_dim_build   = int(ckpt_cfg.get("hidden_dim", hidden_dim_build))
        n_heads_build      = int(ckpt_cfg.get("n_heads", n_heads_build))
        n_layer_build      = int(ckpt_cfg.get("n_layer", n_layer_build))
        dropout_build      = float(ckpt_cfg.get("dropout", dropout_build))
        include_emb_build  = bool(ckpt_cfg.get("include_emb", include_emb_build))
        max_layers_build   = max(max_layers_build, int(ckpt_cfg.get("max_layers", max_layers_build)))
        # If user did not force a rank, inherit from ckpt
        if args.rank is None and "rank" in ckpt_cfg:
            r_model = int(ckpt_cfg["rank"])
            if is_main_process():
                print(f"[hypernet-ckpt] using r_model from checkpoint: {r_model}")
        if ckpt_version is not None:
            hyper_version_to_use = ckpt_version

    # Build HyperNet (with possibly overridden config)
    net = build_hypernet(
        version=hyper_version_to_use,
        text_dim=text_dim_build,
        weight_dim=D,
        rank=r_model,
        hidden_dim=hidden_dim_build,
        n_heads=n_heads_build,
        n_layer=n_layer_build,
        dropout=dropout_build,
        include_emb=include_emb_build,
        module_specs=module_specs,
        max_layers=max_layers_build,
        num_latents=args.num_latents
    ).to(device)

    # Load weights if provided
    if ckpt_state is not None:
        msg = net.load_state_dict(ckpt_state, strict=args.hypernet_strict)
        if is_main_process():
            try:
                print(f"[hypernet-ckpt] load_state_dict(strict={args.hypernet_strict}) "
                      f"-> missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
                if len(msg.missing_keys) > 0:
                    print(f"  first missing keys: {msg.missing_keys[:8]}")
                if len(msg.unexpected_keys) > 0:
                    print(f"  first unexpected keys: {msg.unexpected_keys[:8]}")
            except Exception:
                pass

    # DDP wrap
    if ddp_enabled:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=args.ddp_drop_last) if ddp_enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False if not ddp_enabled else args.ddp_drop_last,
        collate_fn=collate_examples,
    )
    steps_per_epoch = len(loader)
    total_steps = max(1, args.epochs * steps_per_epoch)

    # Resolve warmup
    if args.warmup_steps is not None:
        warmup_steps = int(args.warmup_steps)
    else:
        warmup_steps = int(args.warmup_ratio * total_steps)

    # Scheduler (or None)
    if args.lr_schedule == "cosine":
        scheduler = build_cosine_warmup_scheduler(
            optimizer=optimizer,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    else:
        scheduler = build_constant_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

    # If a legacy checkpoint has no recorded step, infer global_step = epoch * steps_per_epoch.
    if is_resume and (resume_global_step_from_name == 0) and (resume_epoch_from_name > 0):
        resume_global_step_from_name = resume_epoch_from_name * steps_per_epoch
        if is_main_process():
            print(f"[resume] inferred global_step from epoch: {resume_global_step_from_name} "
                  f"(E={resume_epoch_from_name}, steps/epoch={steps_per_epoch})")

    # Optional manual override.
    if is_resume and (args.resume_global_step is not None):
        resume_global_step_from_name = int(args.resume_global_step)
        if is_main_process():
            print(f"[resume] using user-provided --resume-global-step={resume_global_step_from_name}")

    # If this is a resume and the checkpoint carried training state, load it.
    if is_resume:
        # Rebuild the scheduler from saved meta before loading its state (for exact continuity).
        sched_meta = ckpt_extra.get("scheduler_meta")
        if sched_meta is not None:
            if str(sched_meta.get("type", args.lr_schedule)).lower() == "cosine":
                scheduler = build_cosine_warmup_scheduler(
                    optimizer=optimizer,
                    base_lr=float(sched_meta.get("base_lr", args.lr)),
                    min_lr=float(sched_meta.get("min_lr", args.min_lr)),
                    warmup_steps=int(sched_meta.get("warmup_steps", warmup_steps)),
                    total_steps=int(sched_meta.get("total_steps", total_steps)),
                )
            else:
                scheduler = build_constant_warmup_scheduler(
                    optimizer=optimizer,
                    warmup_steps=int(sched_meta.get("warmup_steps", warmup_steps)),
                    total_steps=int(sched_meta.get("total_steps", total_steps)),
                )

        if "optimizer" in ckpt_extra:
            try:
                optimizer.load_state_dict(ckpt_extra["optimizer"])
                if is_main_process():
                    print("[resume] optimizer state loaded.")
            except Exception as e:
                if is_main_process():
                    print(f"[resume] optimizer state load failed: {e}")
        if scheduler is not None and "scheduler" in ckpt_extra and ckpt_extra["scheduler"] is not None:
            try:
                scheduler.load_state_dict(ckpt_extra["scheduler"])
                if is_main_process():
                    print("[resume] scheduler state loaded.")
            except Exception as e:
                if is_main_process():
                    print(f"[resume] scheduler state load failed: {e}")

        # If scheduler state wasn't saved, still position the scheduler & LR to the right stage.
        if (("scheduler" not in ckpt_extra) or (ckpt_extra["scheduler"] is None)) and (scheduler is not None):
            resume_step = int(resume_global_step_from_name)
            scheduler.last_epoch = resume_step - 1  # next .step() -> resume_step
            # Set optimizer LR for the first iteration to the schedule value at resume_step.
            def _lr_factor(schedule_type: str, base_lr: float, min_lr: float,
                           warmup_steps: int, total_steps: int, step: int) -> float:
                schedule_type = schedule_type.lower()
                if schedule_type == "cosine":
                    min_factor = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0
                    if warmup_steps > 0 and step < warmup_steps:
                        return float(step + 1) / float(warmup_steps)
                    if total_steps <= warmup_steps + 1:
                        return 1.0
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
                    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
                else:
                    return float(step + 1) / float(warmup_steps) if (warmup_steps > 0 and step < warmup_steps) else 1.0
            factor = _lr_factor(args.lr_schedule, args.lr, args.min_lr, warmup_steps, total_steps, resume_step)
            optimizer.param_groups[0]["lr"] = args.lr * factor
            if is_main_process():
                print(f"[resume] scheduler positioned at step={resume_step} (factor={factor:.6f}, lr={optimizer.param_groups[0]['lr']:.6e})")

    # W&B init (rank-0 only)
    use_wandb = False
    if args.wandb and is_main_process():
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                id=args.wandb_run_id,
                resume="allow" if args.wandb_run_id else None,
                config={
                    "precision": args.precision,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "text_dim": text_dim_build,
                    "text_len": args.text_len,
                    "hyper_version": hyper_version_to_use,
                    "hidden_dim": hidden_dim_build,
                    "n_heads": n_heads_build,
                    "n_layer": n_layer_build,
                    "dropout": dropout_build,
                    "ddp_world_size": world_size if ddp_enabled else 1,
                    "lr_schedule": args.lr_schedule,
                    "min_lr": args.min_lr,
                    "warmup_steps": warmup_steps,
                    "total_steps": total_steps,
                },
            )
            use_wandb = True
        except ImportError:
            print("[W&B] wandb not installed; continuing without logging.")

    # Text embeddings
    text_embeds_by_lora: Optional[List[torch.Tensor]] = None
    masks_by_lora: Optional[List[torch.Tensor]] = None
    text_bank_multi: Optional[torch.Tensor] = None
    mask_bank_multi: Optional[torch.Tensor] = None
    offsets_cpu: Optional[torch.Tensor] = None

    if args.text_embed_bank is not None:
        if not os.path.exists(args.text_embed_bank):
            raise FileNotFoundError(f"--text-embed-bank not found: {args.text_embed_bank}")
        if is_main_process():
            print(f"[text] Loading text bank from {args.text_embed_bank} (mmap=True)...")
        try:
            # Pass weights_only=False to suppress warning (we trust our own file)
            bank = torch.load(args.text_embed_bank, map_location="cpu", mmap=True, weights_only=False)
        except Exception as e:
            print(f"[text] Warning: mmap load failed ({e}); falling back to standard load (High RAM usage).")
            bank = torch.load(args.text_embed_bank, map_location="cpu", weights_only=False)

        embeds = bank.get("embeddings", None)
        masks = bank.get("masks", None)
        offsets_raw = bank.get("offsets", None)

        # Free the bank dict (the 'prompts'/'names' string lists can be large) before
        # any further processing.
        del bank
        gc.collect()

        if embeds is None:
            raise ValueError(f"--text-embed-bank missing 'embeddings' tensor: {args.text_embed_bank}")
        if embeds.ndim != 3:
            raise ValueError(f"--text-embed-bank 'embeddings' must be [*, T, E], got {tuple(embeds.shape)}")

        T_bank = embeds.shape[1]
        E_bank = embeds.shape[2]
        if E_bank != args.text_dim:
            raise ValueError(f"text_dim mismatch: bank E={E_bank} vs --text-dim={args.text_dim}. Re-encode or change --text-dim.")

        # Handle length mismatch (truncation/padding)
        if T_bank < args.text_len:
            pad = torch.zeros(embeds.shape[0], args.text_len - T_bank, E_bank, dtype=embeds.dtype)
            embeds = torch.cat([embeds, pad], dim=1)
        elif T_bank > args.text_len:
            embeds = embeds[:, :args.text_len, :]

        if masks is not None:
            T_mask = masks.shape[1]
            if T_mask < args.text_len:
                pad_m = torch.ones(masks.shape[0], args.text_len - T_mask, dtype=torch.bool)
                masks = torch.cat([masks, pad_m], dim=1)
            elif T_mask > args.text_len:
                masks = masks[:, :args.text_len]

        if offsets_raw is not None:
            offsets_cpu = offsets_raw.long()
            if offsets_cpu.ndim != 1 or offsets_cpu.numel() < 2:
                raise ValueError(f"Invalid 'offsets' in text bank: shape={tuple(offsets_cpu.shape)}")
            L_from_offsets = int(offsets_cpu.numel() - 1)
            if L_from_offsets != num_loras:
                raise ValueError(
                    f"LoRA count mismatch: bank offsets imply L={L_from_offsets}, but dataset has num_loras={num_loras}."
                )
            text_bank_multi = embeds
            mask_bank_multi = masks
            text_embeds_by_lora = None
            if is_main_process():
                total_prompts = int(text_bank_multi.shape[0])
                print(f"[text] Loaded MULTI bank from {args.text_embed_bank}: "
                      f"N_total={total_prompts}, L={L_from_offsets}")
        else:
            L = embeds.shape[0]
            if L < num_loras:
                raise ValueError(
                    f"Bank has {L} embeddings but dataset has {num_loras} LoRAs. "
                    f"effect_prompts.txt must list >= num_loras prompts in the SAME order as the LoRAs."
                )
            # Legacy path: slice first N embeddings
            # Careful: we cannot index mmapped tensor efficiently into list without copying.
            # We keep the big tensor and slice dynamically or just copy now.
            # Copying 70 embeddings is cheap.
            embeds_slice = embeds[:num_loras]
            text_embeds_by_lora = [embeds_slice[i].clone().to(device=device, dtype=torch.float32) for i in range(num_loras)]
            if masks is not None:
                masks_slice = masks[:num_loras].contiguous()
                masks_by_lora = [
                    masks_slice[i].clone().to(device=device, dtype=torch.bool)
                    for i in range(num_loras)
                ]
            else:
                masks_by_lora = None

            text_bank_multi = None
            mask_bank_multi = None
            offsets_cpu = None
            if is_main_process():
                print(f"[text] Loaded LEGACY bank from {args.text_embed_bank}; using {num_loras} x [{args.text_len}, {args.text_dim}]")
    else:
        text_embeds_by_lora = [
            torch.ones(args.text_len, args.text_dim, device=device, dtype=torch.float32) * float(i + 1)
            for i in range(num_loras)
        ]
        text_bank_multi = None
        offsets_cpu = None

    # Saves model + training state (including scheduler_meta) for exact future resumes.
    def save_checkpoint_fn(epoch_num: int, global_step_now: int):
        if not args.save:
            return
        to_save = net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net
        payload = {
            "hyper_version": hyper_version_to_use,
            "config": {
                "text_dim": text_dim_build,
                "weight_dim": D,
                "rank": r_model,
                "hidden_dim": hidden_dim_build,
                "n_heads": n_heads_build,
                "n_layer": n_layer_build,
                "dropout": dropout_build,
                "module_specs": module_specs,
                "max_layers": max_layers_build,
                "include_emb": include_emb_build,
                "seen_module_types": sorted({ex.module_type for ex in dataset.items}),
                "target_rank": r_target,
                "dataset_rank": r_data,
            },
            "state_dict": to_save.state_dict(),
            # --- training state for true resume ---
            "optimizer": optimizer.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "scheduler_meta": {
                "type": args.lr_schedule,
                "base_lr": args.lr,
                "min_lr": args.min_lr,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
            },
            "epoch": epoch_num,
            "global_step": global_step_now,
        }
        file_name = f"hypernet_epoch_{epoch_num}_step_{global_step_now}.pt"
        save_path = os.path.join(exp_dir, file_name)
        torch.save(payload, save_path)
        print(f"[ckpt] Saved: {save_path}")
        try:
            latest = os.path.join(exp_dir, "latest.pt")
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(save_path), latest)
        except Exception:
            pass

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    # --- initial global step / epoch for resume-or-fresh ---
    global_step = int(resume_global_step_from_name) if is_resume else 0
    start_epoch = int(resume_epoch_from_name + 1) if is_resume else 1

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp_enabled and hasattr(loader, "sampler") and loader.sampler is not None:  # type: ignore[attr-defined]
            loader.sampler.set_epoch(epoch)

        avg_loss, global_step = train_one_epoch(
            net=net,
            loader=loader,
            text_embeds_by_lora=text_embeds_by_lora,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            world_size=world_size,
            log_every=args.log_every,
            log_epoch=args.log_epoch,
            precision=args.precision,
            use_wandb=use_wandb,
            global_step_start=global_step,
            loss_type=args.loss,
            nmse_eps=args.nmse_eps,
            scheduler=scheduler,
            nmse_delta_w=args.nmse_delta_w,
            max_grad_norm=args.max_grad_norm,
            grad_norm_type=args.grad_norm_type,
            num_loras=num_loras,
            ab_rank_mode=args.ab_rank_mode,
            masks_by_lora=masks_by_lora,
            text_bank_multi=text_bank_multi,
            mask_bank_multi=mask_bank_multi,
            offsets=offsets_cpu,
        )
        if is_main_process():
            print(f"[epoch {epoch}] avg loss (global): {avg_loss:.6f}")
            if use_wandb:
                import wandb
                wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch, "step": global_step})

        # Save (rank-0 only) — EPOCH-based saving intact
        if is_main_process() and args.save and epoch % args.save_step == 0:
            save_checkpoint_fn(epoch, global_step)

    # W&B finish
    if use_wandb and is_main_process():
        import wandb
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()