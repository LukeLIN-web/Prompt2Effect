r"""
Example (single GPU, tiny smoke run):

    python train_hypernet_e2e.py \
      --base-model Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
      --hypernet-ckpt ./results/hypernet_mse_new/wan_new/latest.pt \
      --text-embed-bank ./results/summary_json/wan_49effects_text_bank.pt \
      --data-json ./data/effects_train.json \
      --height 480 --width 832 --num-frames 81 \
      --lr 1e-5 --epochs 5 --batch-size 1 --grad-accum 4 \
      --precision bf16 --save ./results/hypernet_e2e/wan_e2e --save-step 1

Multi-GPU (DDP):

    torchrun --nproc_per_node=8 train_hypernet_e2e.py --distributed [same flags...]

`--data-json` is a JSON list of records, e.g.

    [{"video": "/path/clip0.mp4", "image": "/path/clip0_frame0.png", "effect": "squish"},
     {"video": "/path/clip1.mp4", "effect": "inflate"}]    # "image" optional -> uses video frame 0
"""
import os
import gc
import json
import argparse
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# --- reuse the Stage-1 building blocks so the two stages stay in lock-step ---
from train_hypernet_regress import (
    build_hypernet,
    load_hypernet_checkpoint,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    _dtype_for_precision,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =====================================================================================
# Module mapping: diffusers WanTransformer3DModel attention Linear -> Stage-1 module_type
# =====================================================================================
# Stage-1 (the documented Wan run) used the "original Wan" naming, i.e. module types like
# self_attn.q / cross_attn.k_img. The diffusers transformer exposes the same projections under
# attn1 (self) / attn2 (cross). This maps the diffusers sub-path -> the Stage-1 vocab string.
DIFFUSERS_TO_HYPERNET: Dict[Tuple[str, str], str] = {
    ("attn1", "to_q"):       "self_attn.q",
    ("attn1", "to_k"):       "self_attn.k",
    ("attn1", "to_v"):       "self_attn.v",
    ("attn1", "to_out.0"):   "self_attn.o",
    ("attn2", "to_q"):       "cross_attn.q",
    ("attn2", "to_k"):       "cross_attn.k",
    ("attn2", "to_v"):       "cross_attn.v",
    ("attn2", "to_out.0"):   "cross_attn.o",
    ("attn2", "add_k_proj"): "cross_attn.k_img",   # I2V image key/value projections
    ("attn2", "add_v_proj"): "cross_attn.v_img",
}


class LoRATarget:
    """A single Wan Linear we attach a hypernet-predicted LoRA to."""
    __slots__ = ("name", "module", "module_type", "layer_index")

    def __init__(self, name: str, module: nn.Linear, module_type: str, layer_index: int):
        self.name = name
        self.module = module
        self.module_type = module_type
        self.layer_index = layer_index


def discover_lora_targets(
    transformer: nn.Module,
    hypernet_vocab: List[str],
    strict: bool = True,
) -> List[LoRATarget]:
    """
    Walk transformer.blocks and collect the square attention Linear layers the HyperNet can
    predict for, mapping each to its Stage-1 `module_type` and `layer_index` (= block index).
    """
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise AttributeError("Expected `transformer.blocks` (diffusers WanTransformer3DModel).")

    vocab = set(hypernet_vocab)
    targets: List[LoRATarget] = []
    skipped: List[str] = []

    def _get(block, attn_attr, lin_path) -> Optional[nn.Linear]:
        attn = getattr(block, attn_attr, None)
        if attn is None:
            return None
        obj: Any = attn
        for part in lin_path.split("."):
            if part.isdigit():
                obj = obj[int(part)] if obj is not None else None
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj if isinstance(obj, nn.Linear) else None

    for layer_index, block in enumerate(blocks):
        for (attn_attr, lin_path), module_type in DIFFUSERS_TO_HYPERNET.items():
            lin = _get(block, attn_attr, lin_path)
            if lin is None:
                continue
            out_f, in_f = lin.weight.shape
            if out_f != in_f:                       # HyperNet only handles square weights
                skipped.append(f"{module_type}[L{layer_index}] (non-square {out_f}x{in_f})")
                continue
            if module_type not in vocab:
                msg = f"{module_type}[L{layer_index}] not in HyperNet vocab"
                if strict:
                    raise ValueError(
                        f"[module-map] {msg}. HyperNet knows {sorted(vocab)}. "
                        f"Edit DIFFUSERS_TO_HYPERNET or pass --no-strict-module-map."
                    )
                skipped.append(msg)
                continue
            name = f"blocks.{layer_index}.{attn_attr}.{lin_path}"
            targets.append(LoRATarget(name, lin, module_type, layer_index))

    if is_main_process():
        print(f"[module-map] attached {len(targets)} LoRA targets across {len(blocks)} blocks.")
        if skipped:
            print(f"[module-map] skipped {len(skipped)} (first 6): {skipped[:6]}")
    if not targets:
        raise RuntimeError("[module-map] No LoRA targets discovered. Check the transformer/vocab.")
    return targets


# =====================================================================================
# Differentiable LoRA injection
# =====================================================================================
class LoRAInjector:
    """
    Registers a forward hook on each target Linear that adds the HyperNet-predicted update:

        y += scale * (x @ A) @ B          with A:[D, r], B:[r, D]

    The (A, B) tensors are produced by the HyperNet *each step* and kept in the autograd graph,
    so the diffusion loss back-propagates into the HyperNet. Call `set_predictions` before the
    transformer forward and `clear` after the optimizer step.
    """
    def __init__(self, targets: List[LoRATarget], scale: float):
        self.scale = float(scale)
        self.enabled = False
        self._preds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._handles = []
        for t in targets:
            self._handles.append(t.module.register_forward_hook(self._make_hook(t.name)))

    def _make_hook(self, name: str):
        def hook(_module, inputs, output):
            if not self.enabled:
                return output
            ab = self._preds.get(name)
            if ab is None:
                return output
            A, B = ab
            x = inputs[0]
            delta = torch.matmul(torch.matmul(x.to(A.dtype), A), B)  # (x @ A) @ B -> [*, D]
            return output + self.scale * delta.to(output.dtype)
        return hook

    def set_predictions(self, preds: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        self._preds = preds

    def clear(self):
        self._preds = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def _stack_base_weights(targets: List[LoRATarget], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Materialize [n, D, D] base weights for a chunk of targets (no grad: frozen weights)."""
    return torch.stack([t.module.weight.detach().to(device=device, dtype=dtype) for t in targets], dim=0)


def predict_loras(
    hypernet: nn.Module,
    text_embed: torch.Tensor,                 # [1, T, E]
    targets: List[LoRATarget],
    dtype: torch.dtype,
    device: torch.device,
    chunk: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Run the HyperNet over all target modules (chunked to bound memory) and return a
    name -> (A[D,r], B[r,D]) dict, with A/B kept in the autograd graph.
    """
    preds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for start in range(0, len(targets), chunk):
        grp = targets[start:start + chunk]
        base_weight = _stack_base_weights(grp, dtype, device)              # [n, D, D]
        module_type = [t.module_type for t in grp]                        # list[str]
        layer_index = torch.tensor([t.layer_index for t in grp], dtype=torch.long, device=device)
        out = hypernet(
            text_embed,                                                   # [1, T, E] -> cartesian expand to n
            base_weight=base_weight,
            module_type=module_type,
            layer_index=layer_index,
            text_key_padding_mask=None,
        )
        A = out["Lora_A"]   # [n, D, r]
        B = out["Lora_B"]   # [n, r, D]
        for i, t in enumerate(grp):
            preds[t.name] = (A[i], B[i])
    return preds


# =====================================================================================
# Wan I2V forward helpers (mirrors diffusers WanImageToVideoPipeline)
# =====================================================================================
def _vae_norm_stats(vae, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_video_to_latents(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels: [B,3,F,H,W] in [-1,1] -> normalized latents [B,z,F',h,w]."""
    mean, std = _vae_norm_stats(vae, pixels.device, pixels.dtype)
    z = vae.encode(pixels).latent_dist.sample()
    return (z - mean) * std


@torch.no_grad()
def build_i2v_condition(vae, cond_image: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    cond_image: [B,3,H,W] in [-1,1] (the first frame). Returns the I2V condition tensor
    [B, t_comp + z, F', h, w] = concat(mask, encoded-first-frame-latents) along channels,
    matching diffusers WanImageToVideoPipeline.prepare_latents.
    """
    B, C, H, W = cond_image.shape
    t_comp = vae.config.scale_factor_temporal if hasattr(vae.config, "scale_factor_temporal") else 4
    s_comp = vae.config.scale_factor_spatial if hasattr(vae.config, "scale_factor_spatial") else 8
    h, w = H // s_comp, W // s_comp

    # First frame followed by zeros for the remaining (num_frames - 1) frames.
    video_condition = torch.cat(
        [cond_image.unsqueeze(2), cond_image.new_zeros(B, C, num_frames - 1, H, W)], dim=2
    )
    mean, std = _vae_norm_stats(vae, cond_image.device, cond_image.dtype)
    latent_condition = vae.encode(video_condition).latent_dist.mode()
    latent_condition = (latent_condition - mean) * std                    # [B,z,F',h,w]

    # Mask: 1 on the (single) conditioning frame, 0 elsewhere, reshaped to t_comp channels.
    mask = torch.ones(B, 1, num_frames, h, w, device=cond_image.device, dtype=latent_condition.dtype)
    mask[:, :, 1:] = 0
    first = mask[:, :, :1].repeat_interleave(t_comp, dim=2)               # expand the first frame
    mask = torch.cat([first, mask[:, :, 1:]], dim=2)                      # [B,1,num_frames+t_comp-1,h,w]
    mask = mask.view(B, -1, t_comp, h, w).transpose(1, 2)                 # [B,t_comp,F',h,w]
    return torch.cat([mask, latent_condition], dim=1)                     # [B,t_comp+z,F',h,w]


@torch.no_grad()
def encode_clip_image(image_encoder, image_processor, pil_image) -> torch.Tensor:
    """CLIP image embeds for I2V: penultimate hidden state, as diffusers does."""
    inputs = image_processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(image_encoder.device, dtype=image_encoder.dtype)
    out = image_encoder(pixel_values=pixel_values, output_hidden_states=True)
    return out.hidden_states[-2]


def sample_sigmas(batch: int, device: torch.device, mode: str, shift: float) -> torch.Tensor:
    """Flow-matching noise level in (0,1), with the Wan resolution `shift` applied."""
    if mode == "uniform":
        sigma = torch.rand(batch, device=device)
    else:  # logit_normal (SD3/Wan-style)
        sigma = torch.sigmoid(torch.randn(batch, device=device))
    if shift and shift != 1.0:
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return sigma.clamp(1e-4, 1 - 1e-4)


# =====================================================================================
# Dataset
# =====================================================================================
def _load_video_frames(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """
    Decode `num_frames` frames and return pixels [3, F, H, W] in [-1, 1].
    Uses decord if available, else torchvision.io. Replace with your project loader as needed.
    """
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path, height=height, width=width)
        idx = torch.linspace(0, max(len(vr) - 1, 0), num_frames).round().long().tolist()
        frames = vr.get_batch(idx)                       # [F,H,W,3] uint8
    except Exception:
        from torchvision.io import read_video
        import torchvision.transforms.functional as TF
        vid, _, _ = read_video(path, output_format="THWC", pts_unit="sec")  # [T,H,W,3]
        if vid.shape[0] == 0:
            raise RuntimeError(f"No frames decoded from {path}")
        idx = torch.linspace(0, vid.shape[0] - 1, num_frames).round().long()
        frames = vid[idx]
        frames = TF.resize(frames.permute(0, 3, 1, 2), [height, width]).permute(0, 2, 3, 1)

    pixels = frames.float().permute(3, 0, 1, 2) / 127.5 - 1.0             # [3,F,H,W] in [-1,1]
    return pixels.contiguous()


class VideoEffectDataset(Dataset):
    """Records: {"video": path, "image": path?, "effect": name}. Returns paths + effect idx."""
    def __init__(self, data_json: str, effect_to_idx: Dict[str, int]):
        with open(data_json, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.effect_to_idx = effect_to_idx
        missing = sorted({r["effect"] for r in self.records if r["effect"] not in effect_to_idx})
        if missing:
            raise ValueError(f"[data] effects not in text-embed bank: {missing[:10]} ...")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {
            "video": r["video"],
            "image": r.get("image"),                     # may be None -> use frame 0
            "effect": r["effect"],
            "effect_idx": self.effect_to_idx[r["effect"]],
        }


def collate(batch):
    # batch_size is expected to be 1 effect per forward (one global injection per step).
    return batch


def load_text_bank(path: str, device: torch.device, dtype: torch.dtype):
    """Load the encode_prompts.py bank -> (embeddings [L,T,E] on CPU, name->idx)."""
    bank = torch.load(path, map_location="cpu", weights_only=False)
    names = bank["names"]
    embeds = bank["embeddings"]                           # [L, T, E]
    name_to_idx = {n: i for i, n in enumerate(names)}
    if is_main_process():
        print(f"[text] bank '{path}': L={len(names)} T={embeds.shape[1]} E={embeds.shape[2]}")
    return embeds, name_to_idx


# =====================================================================================
# Model loading
# =====================================================================================
def load_wan_components(base_model: str, dtype: torch.dtype, device: torch.device,
                        grad_ckpt: bool = True):
    """Load frozen VAE + CLIP image encoder + transformer + scheduler from a diffusers Wan I2V repo."""
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    if is_main_process():
        print(f"[wan] loading components from {base_model} ...")
    vae = AutoencoderKLWan.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(base_model, vae=vae, torch_dtype=dtype)

    transformer = pipe.transformer
    image_encoder = pipe.image_encoder
    image_processor = pipe.image_processor
    scheduler = pipe.scheduler

    for m in (vae, image_encoder, transformer):
        m.requires_grad_(False)
        m.eval()
    vae.to(device)
    image_encoder.to(device)
    transformer.to(device)
    if grad_ckpt and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    return vae, image_encoder, image_processor, transformer, scheduler


def build_hypernet_from_ckpt(ckpt_path: str, device: torch.device):
    """Rebuild the Stage-1 HyperNet architecture from its checkpoint config and load weights."""
    version, cfg, state, _extra = load_hypernet_checkpoint(ckpt_path)
    if cfg is None:
        raise ValueError(f"[hypernet] checkpoint {ckpt_path} has no 'config'; cannot rebuild architecture.")
    version = version or "ve"
    module_specs = cfg.get("module_specs", None)
    if not module_specs:
        raise ValueError("[hypernet] checkpoint config missing 'module_specs' (the module vocabulary).")

    net = build_hypernet(
        version=version,
        text_dim=int(cfg["text_dim"]),
        weight_dim=int(cfg["weight_dim"]),
        rank=int(cfg["rank"]),
        hidden_dim=int(cfg["hidden_dim"]),
        n_heads=int(cfg["n_heads"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg.get("dropout", 0.1)),
        include_emb=bool(cfg.get("include_emb", True)),
        module_specs=module_specs,
        max_layers=int(cfg.get("max_layers", 32)),
        num_latents=int(cfg.get("num_latents", 256)),
    ).to(device)

    msg = net.load_state_dict(state, strict=False)
    if is_main_process():
        print(f"[hypernet] init from {ckpt_path} (version={version}, rank={cfg['rank']}, "
              f"D={cfg['weight_dim']}); missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    hypernet_vocab = list(module_specs.keys())
    return net, hypernet_vocab, cfg, version


def save_checkpoint(net, optimizer, cfg, version, epoch, global_step, save_dir):
    """Save in the SAME format as Stage-1 so e2e_wan_infer.py / further training can load it."""
    os.makedirs(save_dir, exist_ok=True)
    to_save = net.module if isinstance(net, nn.parallel.DistributedDataParallel) else net
    payload = {
        "hyper_version": version,
        "config": cfg,
        "state_dict": to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "stage": "e2e",
    }
    path = os.path.join(save_dir, f"hypernet_e2e_epoch_{epoch}_step_{global_step}.pt")
    torch.save(payload, path)
    print(f"[ckpt] saved {path}")
    try:
        latest = os.path.join(save_dir, "latest.pt")
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except Exception:
        pass


# =====================================================================================
# Training
# =====================================================================================
def parse_args():
    p = argparse.ArgumentParser("Stage-2 end-to-end (diffusion-loss) HyperNet training on Wan2.1 I2V.")
    # models / data
    p.add_argument("--base-model", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    p.add_argument("--hypernet-ckpt", type=str, required=True, help="Stage-1 regress checkpoint (initialization).")
    p.add_argument("--text-embed-bank", type=str, required=True, help=".pt from encode_prompts.py (names+embeddings).")
    p.add_argument("--data-json", type=str, required=True, help="JSON list of {video, image?, effect} records.")
    p.add_argument("--save", type=str, default=None, help="Directory to save HyperNet checkpoints.")
    p.add_argument("--save-step", type=int, default=1, help="Save every N epochs.")
    # video / latent geometry
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    # optimization
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1, help="Clips per step; all must share ONE effect.")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--num-workers", type=int, default=2)
    # lora injection / hypernet
    p.add_argument("--lora-scale", type=float, default=1.0, help="Match the teacher LoRA alpha/rank convention.")
    p.add_argument("--hypernet-chunk", type=int, default=16, help="Modules per HyperNet forward (memory knob).")
    p.add_argument("--no-strict-module-map", action="store_true", help="Warn instead of erroring on unmapped modules.")
    p.add_argument("--hypernet-grad-ckpt", action="store_true", help="Gradient-checkpoint the HyperNet internals.")
    # flow matching
    p.add_argument("--timestep-sampling", type=str, default="logit_normal", choices=["uniform", "logit_normal"])
    p.add_argument("--flow-shift", type=float, default=3.0, help="Wan resolution shift (3.0 for 480p, 5.0 for 720p).")
    # logging / ddp
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device, rank, world_size, ddp = setup_distributed(args.distributed, args.device)
    torch.manual_seed(args.seed + rank)
    dtype = _dtype_for_precision(args.precision)

    # --- HyperNet (trainable) initialized from Stage-1 ---
    hypernet, hypernet_vocab, hyper_cfg, hyper_version = build_hypernet_from_ckpt(args.hypernet_ckpt, device)
    if args.hypernet_grad_ckpt and hasattr(hypernet, "grad_ckpt"):
        hypernet.grad_ckpt = True
    hypernet.train()

    # --- Frozen Wan components ---
    vae, image_encoder, image_processor, transformer, scheduler = load_wan_components(
        args.base_model, dtype, device, grad_ckpt=True
    )

    # --- LoRA targets + injector ---
    targets = discover_lora_targets(transformer, hypernet_vocab, strict=not args.no_strict_module_map)
    injector = LoRAInjector(targets, scale=args.lora_scale)

    # sanity: HyperNet weight_dim must match the transformer's attention width
    D_model = targets[0].module.weight.shape[0]
    if int(hyper_cfg["weight_dim"]) != int(D_model):
        raise ValueError(f"[check] HyperNet weight_dim={hyper_cfg['weight_dim']} != transformer width {D_model}.")

    # --- text bank + data ---
    text_embeds, name_to_idx = load_text_bank(args.text_embed_bank, device, dtype)
    dataset = VideoEffectDataset(args.data_json, name_to_idx)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=collate, drop_last=False,
    )

    # --- DDP wrap + optimizer (HyperNet only) ---
    if ddp:
        hypernet = nn.parallel.DistributedDataParallel(
            hypernet, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,    # not every module_type fires every step
        )
    hn_module = hypernet.module if isinstance(hypernet, nn.parallel.DistributedDataParallel) else hypernet
    optimizer = torch.optim.AdamW(hn_module.parameters(), lr=args.lr)

    def autocast_ctx():
        if args.precision == "bf16" and device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    use_wandb = False
    if args.wandb and is_main_process():
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       config={**vars(args), "n_targets": len(targets)})
            use_wandb = True
        except ImportError:
            print("[wandb] not installed; continuing without logging.")

    global_step = 0
    num_latent_frames = (args.num_frames - 1) // (
        vae.config.scale_factor_temporal if hasattr(vae.config, "scale_factor_temporal") else 4
    ) + 1
    if is_main_process():
        print(f"[train] {len(dataset)} clips | {len(targets)} LoRA targets | "
              f"latent frames={num_latent_frames} | rank={hyper_cfg['rank']} | dtype={dtype}")

    for epoch in range(1, args.epochs + 1):
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(loader):
            # one global injection per step => the microbatch must share a single effect
            effects = {b["effect"] for b in batch}
            if len(effects) != 1:
                raise RuntimeError(f"[train] step {it}: microbatch mixes effects {effects}; "
                                   f"use batch-size 1 or group clips by effect.")
            rec0 = batch[0]
            effect_idx = rec0["effect_idx"]
            text_embed = text_embeds[effect_idx:effect_idx + 1].to(device=device, dtype=dtype)  # [1,T,E]

            # ---- build the diffusion inputs (frozen, no grad) ----
            pixels = torch.stack(
                [_load_video_frames(b["video"], args.num_frames, args.height, args.width) for b in batch],
                dim=0,
            ).to(device=device, dtype=torch.float32)               # [B,3,F,H,W] in [-1,1]

            # Conditioning image: an explicit "image" overrides the video's first frame.
            cond_image = pixels[:, :, 0].clone()                   # [B,3,H,W]
            if rec0["image"]:
                from diffusers.utils import load_image
                import torchvision.transforms.functional as TF
                img = load_image(rec0["image"]).resize((args.width, args.height))
                cond_image[0] = (TF.to_tensor(img).to(device) * 2.0 - 1.0)

            with torch.no_grad():
                x0 = encode_video_to_latents(vae, pixels).to(dtype)
                condition = build_i2v_condition(vae, cond_image.float(), args.num_frames).to(dtype)
                # CLIP image embeds expect a PIL image; use the first sample's conditioning frame.
                from PIL import Image
                first_np = ((cond_image[0].permute(1, 2, 0).float().cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy()
                image_embeds = encode_clip_image(image_encoder, image_processor, Image.fromarray(first_np)).to(dtype)

            # ---- HyperNet predicts LoRA for every target module (chunked, differentiable) ----
            with autocast_ctx():
                preds = predict_loras(hn_module, text_embed, targets, dtype, device, args.hypernet_chunk)
                injector.set_predictions(preds)
                injector.enabled = True

                # ---- flow-matching diffusion loss ----
                noise = torch.randn_like(x0)
                sigma = sample_sigmas(x0.shape[0], device, args.timestep_sampling, args.flow_shift)
                sig = sigma.view(-1, 1, 1, 1, 1)
                noisy = (1.0 - sig) * x0 + sig * noise
                target = noise - x0
                timestep = (sigma * 1000.0).to(dtype)
                model_input = torch.cat([noisy, condition], dim=1)

                model_pred = transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=text_embed.expand(x0.shape[0], -1, -1),
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float()) / args.grad_accum

            loss.backward()
            injector.enabled = False
            injector.clear()

            if (it + 1) % args.grad_accum == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(hn_module.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main_process() and (global_step % args.log_every == 0):
                    lv = loss.item() * args.grad_accum
                    print(f"[epoch {epoch} step {global_step}] effect={rec0['effect']} loss={lv:.5f}")
                    if use_wandb:
                        import wandb
                        wandb.log({"train/loss": lv, "epoch": epoch, "step": global_step})

            del pixels, x0, condition, image_embeds, preds
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if is_main_process() and args.save and (epoch % args.save_step == 0):
            save_checkpoint(hn_module, optimizer, hyper_cfg, hyper_version, epoch, global_step, args.save)

    injector.remove()
    if use_wandb and is_main_process():
        import wandb
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
