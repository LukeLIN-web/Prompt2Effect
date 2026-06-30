#!/usr/bin/env bash
# Stage-2: end-to-end (diffusion-loss) fine-tuning of the HyperNet on Wan2.1 I2V,
# initialized from the Stage-1 regress checkpoint. See train_hypernet_e2e.py for caveats.
set -euo pipefail

export PYTHONUNBUFFERED=1

BASE_MODEL=${BASE_MODEL:-Wan-AI/Wan2.1-I2V-14B-480P-Diffusers}
HYPERNET_CKPT=${HYPERNET_CKPT:-./results/hypernet_regress/wan/latest.pt}  # Stage-1 init
TEXT_BANK=${TEXT_BANK:-./results/wan_effects_text_bank.pt}                # from encode_prompts.py
DATA_JSON=${DATA_JSON:-./data/effects_train.json}                        # [{video, image?, effect}, ...]
SAVE=${SAVE:-./results/hypernet_e2e/wan}

mkdir -p "$(dirname "$SAVE")" logs

# Single node, 8 GPUs. Drop --distributed / torchrun for a 1-GPU debug run.
torchrun --max-restarts=0 --nnodes 1 --nproc_per_node=8 \
    train_hypernet_e2e.py --distributed \
    --base-model "$BASE_MODEL" \
    --hypernet-ckpt "$HYPERNET_CKPT" \
    --text-embed-bank "$TEXT_BANK" \
    --data-json "$DATA_JSON" \
    --save "$SAVE" --save-step 1 \
    --height 480 --width 832 --num-frames 81 \
    --lr 1e-5 --epochs 10 --batch-size 1 --grad-accum 4 --precision bf16 \
    --lora-scale 1.0 --hypernet-chunk 16 --hypernet-grad-ckpt \
    --timestep-sampling logit_normal --flow-shift 3.0 --log-every 10 \
    2>&1 | tee -a logs/train_hypernet_e2e.log
