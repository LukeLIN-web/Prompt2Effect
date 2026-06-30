import argparse
import os
import json
import re
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import numpy as np
import torch
from transformers import CLIPVisionModel
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# NEW IMPORTS for LoRA layer filtering
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# -----------------------------
# Utilities
# -----------------------------
def _slugify(text: str, max_len: int = 64) -> str:
    """Create a filesystem-friendly slug from a prompt or filename."""
    if not text:
        return "item"
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^A-Za-z0-9._-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    if not text:
        text = "item"
    return text[:max_len]

def _read_json_records(path: Optional[str]) -> List[Dict[str, Any]]:
    """Read either a JSON array of objects or JSON Lines."""
    if not path or not os.path.exists(path):
        return []
        
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("--input_json must be a list of objects.")
        return data
        
    # JSONL fallback
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {ln} in {path} is not a JSON object.")
            recs.append(obj)
    return recs

def _pick_prompt(rec: Dict[str, Any], prompt_key: str = "prompt") -> Optional[str]:
    def _get(d: Dict[str, Any], k: str) -> Optional[str]:
        v = d.get(k)
        return v if isinstance(v, str) and v.strip() else None

    p = _get(rec, prompt_key) or _get(rec, "caption")
    if p: return p

    per = rec.get("prompt_enhancer_response")
    if isinstance(per, dict):
        v = per.get("final designed prompt")
        if isinstance(v, str) and v.strip():
            return v

    op = rec.get("original_prompt")
    if isinstance(op, str) and op.strip():
        try:
            arr = json.loads(op)
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                v = arr[0].get("prompt")
                if isinstance(v, str) and v.strip():
                    return v
        except Exception:
            return op
    return None

def _resolve_image_path(val: str, json_root: Optional[str]) -> str:
    """Prepend json_root for relative paths; leave absolute paths untouched."""
    if not isinstance(val, str):
        raise ValueError("JSON 'image' value must be a string path.")
    if os.path.isabs(val) or val.startswith(("http://", "https://", "s3://")):
        return val
    if json_root:
        return os.path.normpath(os.path.join(json_root, val))
    return val

def _should_skip(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False

def run_garbage_collection():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@contextmanager
def benchmark_timing(mode_name="Inference"):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[{mode_name}] Completed in {elapsed:.2f} seconds.")

def resize_for_wan(image, pipe, max_area=480*832):
    """Resizes image to fit Wan2.1's latent patch size requirements."""
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return image.resize((width, height)), width, height

# ---------- ARGUMENTS ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run e2e Wan2.1 Image2Video inference. Supports batch via --input_json.")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    parser.add_argument("--lora_weight", type=str, default="Remade/Squish", help="Path or HuggingFace repo to LoRA weights")
    parser.add_argument("--lora_weight_name", type=str, default=None, help="Name of the safetensors file if loading from a folder/hub.")
    parser.add_argument("--target_layer", type=int, default=None, help="If set, only apply LoRA to this specific layer index (e.g., 12).")
    parser.add_argument("--disable_lora", action="store_true", help="If set, skips loading the LoRA completely and runs the base model.")
    parser.add_argument("--disable_cpu_offload", action="store_true", help="Disable CPU offloading if you have massive VRAM")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--max_area", type=int, default=480*832, help="Max area for image resizing")
    parser.add_argument("--output_dir", type=str, default="results")

    # Single-item args
    parser.add_argument("--prompt", type=str, help="Prompt for single-item mode")
    parser.add_argument("--image_path", type=str, help="Image path or URL for single-item mode")

    # Batch-mode args
    parser.add_argument("--input_json", type=str, default=None,
        help="Path to a JSON list or JSONL of records for batch mode. If omitted, runs single-item mode using --prompt/--image_path.")
    parser.add_argument("--json_image_key", type=str, default="image", help="Key in JSON records for the image path.")
    parser.add_argument("--json_prompt_key", type=str, default="prompt", help="Preferred key for the prompt in JSON records.")
    parser.add_argument("--json_root", type=str, default=None, help="Optional directory to prepend to relative image paths in JSON.")
    
    parser.add_argument("--start_idx", type=int, default=0, help="Start index inside the JSON (inclusive).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of items to process from --start_idx.")
    parser.add_argument("--n_per_item", type=int, default=1, help="Number of variations per record.")
    parser.add_argument("--skip_missing", action="store_true", default=False, help="Skip records whose image file does not exist locally.")
    parser.add_argument("--skip_existing", action="store_true", default=False, help="Skip if the output file already exists.")

    return parser.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading Wan2.1 models from {args.model_id}...")
    
    # Load separate components with correct dtypes
    image_encoder = CLIPVisionModel.from_pretrained(args.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id, 
        vae=vae, 
        image_encoder=image_encoder, 
        torch_dtype=torch.bfloat16
    )

    # --- LORA LOADING LOGIC ---
    if args.lora_weight and not args.disable_lora:
        print(f"Loading LoRA weights: {args.lora_weight}")
        
        if args.target_layer is not None:
            print(f"Filtering LoRA to ONLY apply to layer {args.target_layer}...")
            
            # Resolve the file path (local file, local dir, or HuggingFace Hub)
            if os.path.isfile(args.lora_weight):
                lora_path = args.lora_weight
            else:
                w_name = args.lora_weight_name or "pytorch_lora_weights.safetensors"
                if os.path.isdir(args.lora_weight):
                    lora_path = os.path.join(args.lora_weight, w_name)
                else:
                    # Download from HuggingFace
                    lora_path = hf_hub_download(repo_id=args.lora_weight, filename=w_name)
            
            # Load the raw state dictionary
            state_dict = load_file(lora_path)
            
            # Filter keys containing the specific layer number (e.g., '.12.')
            layer_pattern = f".{args.target_layer}."
            filtered_state_dict = {
                k: v for k, v in state_dict.items() 
                if layer_pattern in k
            }
            
            if not filtered_state_dict:
                print(f"[WARN] No keys found for layer {args.target_layer}. Is the layer index correct?")
            else:
                print(f"Found {len(filtered_state_dict)} tensors matching layer {args.target_layer}.")
            
            # Load the filtered dictionary natively into diffusers
            pipe.load_lora_weights(filtered_state_dict)
            
        else:
            # Standard full LoRA load
            pipe.load_lora_weights(args.lora_weight, weight_name=args.lora_weight_name)
    else:
        print("LoRA loading is disabled. Running base model.")

    if not args.disable_cpu_offload:
        print("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # -----------------------------
    # Batch mode (JSON)
    # -----------------------------
    records = _read_json_records(args.input_json)
    if records:
        if args.limit is not None:
            end = args.start_idx + args.limit
            records = records[args.start_idx:end]
        else:
            records = records[args.start_idx:]
            
        print(f"[BATCH] Loaded {len(records)} records for processing.")

        for idx, rec in enumerate(records, start=args.start_idx):
            img_raw = rec.get(args.json_image_key)
            if img_raw is None:
                print(f"[WARN] Skipping index {idx}: no '{args.json_image_key}' key.")
                continue
            
            image_input_path = _resolve_image_path(str(img_raw), args.json_root)
            prompt = _pick_prompt(rec, prompt_key=args.json_prompt_key)
            
            if not prompt:
                print(f"[WARN] Skipping index {idx}: no usable prompt found.")
                continue

            # Check if local file required
            is_remote = image_input_path.startswith(("http://", "https://", "s3://"))
            if args.skip_missing and not is_remote and not os.path.exists(image_input_path):
                print(f"[WARN] Skipping index {idx}: image not found at {image_input_path}")
                continue

            # Prepare image
            try:
                raw_image = load_image(image_input_path)
                resized_image, width, height = resize_for_wan(raw_image, pipe, max_area=args.max_area)
            except Exception as e:
                print(f"[ERROR] Failed to load/resize image at {image_input_path}: {e}")
                continue

            stem = os.path.basename(image_input_path).split('?')[0]
            base = os.path.splitext(stem)[0]
            slug_base = _slugify(prompt) if prompt else _slugify(base)
            prefix = f"{idx:05d}_{slug_base}"

            for rep in range(max(1, args.n_per_item)):
                seed = int(args.seed + idx * 9973 + rep * 101)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                out_name = f"{prefix}_rep{rep}.mp4"
                out_path = os.path.join(args.output_dir, out_name)

                if args.skip_existing and _should_skip(out_path):
                    print(f"[SKIP] Exists: {out_path}")
                    continue

                print(f"\n[GEN] Index {idx} Rep {rep} | Prompt: {prompt[:60]}...")
                with benchmark_timing("Wan2.1 Generation"):
                    output = pipe(
                        image=resized_image,
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_frames=args.num_frames,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator
                    ).frames[0]
                    
                export_to_video(output, out_path, fps=args.fps)
                
                # Save sidecar metadata
                sidecar = {
                    "index": idx,
                    "rep": rep,
                    "image_input": image_input_path,
                    "prompt": prompt,
                    "seed": seed,
                    "fps": args.fps,
                    "frames": args.num_frames,
                    "guidance_scale": args.guidance_scale,
                    "width": width,
                    "height": height,
                    "lora_used": args.lora_weight if not args.disable_lora else "None",
                    "target_layer": args.target_layer
                }
                with open(os.path.join(args.output_dir, f"{prefix}_rep{rep}.json"), "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2)

                print(f"[OK] Wrote {out_path}")
                run_garbage_collection()

        print(f"\n[DONE] Batch completed. Outputs in: {args.output_dir}")
        return

    # -----------------------------
    # Single-item mode
    # -----------------------------
    if not args.prompt or not args.image_path:
        raise SystemExit("Provide --prompt and --image_path for single-item mode, or ensure your --input_json file exists.")

    print(f"\n[SINGLE] Prompt: {args.prompt}")
    raw_image = load_image(args.image_path)
    resized_image, width, height = resize_for_wan(raw_image, pipe, max_area=args.max_area)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    output_filename = "wan_image2video_output.mp4"
    out_path = os.path.join(args.output_dir, output_filename)

    with benchmark_timing("Wan2.1 Generation"):
        output = pipe(
            image=resized_image,
            prompt=args.prompt,
            height=height,
            width=width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).frames[0]

    export_to_video(output, out_path, fps=args.fps)
    print(f"[DONE] Wrote {out_path}")
    run_garbage_collection()

if __name__ == "__main__":
    main()