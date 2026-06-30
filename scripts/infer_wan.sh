#!/usr/bin/env bash
# Run Wan2.1 I2V inference with a LoRA (single-item or batch via --input_json).
set -euo pipefail

MODEL_ID=${MODEL_ID:-Wan-AI/Wan2.1-I2V-14B-480P-Diffusers}
LORA=${LORA:-Remade/Squish}          # HF repo, local dir, or .safetensors produced by the pipeline
OUTDIR=${OUTDIR:-./results/infer}

# --- single item ---
python e2e_wan_infer.py \
    --model_id "$MODEL_ID" \
    --lora_weight "$LORA" \
    --prompt "A cat sitting on a chair, then it gets squished." \
    --image_path ./assets/cat.png \
    --output_dir "$OUTDIR" \
    --num_frames 81 --fps 16 --guidance_scale 5.0 --num_inference_steps 28

# --- batch over a JSON of records (uncomment) ---
# python e2e_wan_infer.py \
#     --model_id "$MODEL_ID" --lora_weight "$LORA" \
#     --input_json ./wan_eval/cakeify_output.json \
#     --json_image_key image --json_prompt_key prompt \
#     --output_dir "$OUTDIR" --skip_existing
