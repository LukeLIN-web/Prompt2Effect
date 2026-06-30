#!/usr/bin/env bash
# Stage-1: train the HyperNet to regress onto a bank of pre-trained per-effect teacher LoRAs.
# Edit the paths below to point at your base Wan checkpoint, teacher LoRAs, and text-embed bank.
set -euo pipefail

export PYTHONUNBUFFERED=1
# Multi-node NCCL knobs are cluster-specific; uncomment/adapt for your fabric (e.g. EFA):
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth

BASE_CKPT=${BASE_CKPT:-/path/to/Wan2.1-Original}          # base model weights (state_dict / diffusers dir)
LORA_DIR=${LORA_DIR:-/path/to/wan_lora}                   # directory of teacher LoRA .safetensors
TEXT_BANK=${TEXT_BANK:-./results/wan_effects_text_bank.pt} # from encode_prompts.py
SAVE=${SAVE:-./results/hypernet_regress/wan}
DATASET_CACHE=${DATASET_CACHE:-./results/effects_wan.pt}   # built on first run, reused after

mkdir -p "$(dirname "$SAVE")" logs

# Collect every *.safetensors in $LORA_DIR as a teacher LoRA (one --lora-ckpt each).
LORA_ARGS=()
for f in "$LORA_DIR"/*.safetensors; do LORA_ARGS+=(--lora-ckpt "$f"); done

# Single node, 8 GPUs. For multi-node use --nnodes N --node-rank R --rdzv-endpoint HOST:PORT.
torchrun --max-restarts=0 --nnodes 1 --nproc_per_node=8 \
    -m train_hypernet_regress --distributed \
    --base-ckpt "$BASE_CKPT" "${LORA_ARGS[@]}" \
    --dataset-cache "$DATASET_CACHE" \
    --text-embed-bank "$TEXT_BANK" \
    --save "$SAVE" --save_step 500 \
    --device cuda --epochs 40000 --batch-size 18 --precision bf16 \
    --hyper-version ve --hidden-dim 2048 --n-layer 14 --n-heads 32 --include_emb \
    --loss nmse --lr 5e-5 --lr-schedule constant --max-grad-norm 1.0 \
    --log-every 20 --log-epoch 10 \
    --weight-regex "(self_attn|cross_attn|ffn)\.(q|k|v|k_img|v_img|o|0|2)" \
    2>&1 | tee -a logs/train_hypernet_regress.log
