# Prompt2Effect (Wan2.1 reproduction)

Code for **Prompt2Effect** — predicting per-effect LoRA weights for a video diffusion model
directly from a text prompt with a HyperNetwork.

- 📄 Paper: <https://arxiv.org/abs/2606.13971>
- 🌐 Project page: <https://xiaomeng-yang.github.io/Prompt2Effect/>

> **This repository is a reproduction of the pipeline on the public
> [Wan2.1](https://github.com/Wan-Video/Wan2.1) image-to-video model** (via 🤗 `diffusers`),
> using the [Remade-AI Wan2.1-14B-480P I2V effect LoRAs](https://huggingface.co/collections/Remade-AI/wan21-14b-480p-i2v-loras)
> as the per-effect teachers. Numbers may differ from the paper.

## Method at a glance

A HyperNetwork maps an *effect* text prompt to a set of LoRA weight updates `(A, B)` for the
attention projections of the Wan transformer.

1. **Stage-1 — Regression** *(the Prompt2Effect method, reproduced here)*. Train the HyperNet to
   reproduce a bank of pre-trained per-effect *teacher* LoRAs (NMSE/MSE on the LoRA factors).
   Fast and stable. This is the stage described in the paper and the main focus of this repo.
2. **Stage-2 — End-to-end** *(experimental extension — ⚠️ not verified)*. Initialize from
   Stage-1, predict LoRA weights, **inject them into the frozen Wan transformer**, and
   back-propagate the **flow-matching diffusion loss** on real `(video, effect)` data into the
   HyperNet. This goes beyond the original paper: the trainer is a faithful draft that has **not
   been validated end to end** yet.

At inference, the HyperNet produces a LoRA for a new effect prompt that is applied to Wan I2V.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You need access to the Wan2.1 weights on the Hugging Face Hub
(`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`) and a set of per-effect teacher LoRAs for Stage-1 — this
reproduction uses the 51 effect LoRAs from the
[Remade-AI Wan2.1-14B-480P I2V collection](https://huggingface.co/collections/Remade-AI/wan21-14b-480p-i2v-loras).

## Usage

### 0. Encode effect prompts into a text-embedding bank

`effect_prompts.txt` lists one `name: prompt` per effect (the order must match your teacher
LoRA order for Stage-1).

```bash
python encode_prompts.py \
  --effect-prompts ./data/effect_prompts.txt \
  --language-model-ckpt google/t5-v1_1-xxl \
  --output ./results/wan_effects_text_bank.pt
```

### 1. Stage-1: regression training

```bash
bash scripts/train_hypernet_regress.sh
# or directly:
torchrun --nproc_per_node=8 -m train_hypernet_regress --distributed \
  --base-ckpt /path/to/Wan2.1-Original \
  --lora-ckpt /path/to/wan_lora/squish.safetensors  [--lora-ckpt ...] \
  --text-embed-bank ./results/wan_effects_text_bank.pt \
  --hyper-version ve --hidden-dim 2048 --n-layer 14 --n-heads 32 --include_emb \
  --loss nmse --lr 5e-5 --precision bf16 \
  --weight-regex "(self_attn|cross_attn|ffn)\.(q|k|v|k_img|v_img|o|0|2)" \
  --save ./results/hypernet_regress/wan
```

### 2. Stage-2: end-to-end (diffusion-loss) training — ⚠️ experimental / unverified

> This stage is an extension beyond the paper and has **not been verified end to end**. 

Initializes from the Stage-1 checkpoint and fine-tunes through the Wan I2V diffusion loss.
`--data-json` is a list of `{"video": ..., "image": <first frame, optional>, "effect": <name>}`.

```bash
bash scripts/train_hypernet_e2e.sh
# or directly:
python train_hypernet_e2e.py \
  --base-model Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --hypernet-ckpt ./results/hypernet_regress/wan/latest.pt \
  --text-embed-bank ./results/wan_effects_text_bank.pt \
  --data-json ./data/effects_train.json \
  --lr 1e-5 --precision bf16 --hypernet-grad-ckpt \
  --save ./results/hypernet_e2e/wan
```

### 3. Inference

```bash
bash scripts/infer_wan.sh
# or directly (single item):
python e2e_wan_infer.py \
  --model_id Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --lora_weight /path/to/predicted_or_trained_lora.safetensors \
  --prompt "..." --image_path ./assets/input.png \
  --output_dir ./results/infer
```

## Release roadmap

Planned for this reproduction. The teacher LoRAs are the 51 effects from the
[Remade-AI Wan2.1-14B-480P I2V collection](https://huggingface.co/collections/Remade-AI/wan21-14b-480p-i2v-loras).

**Stage-1 (regression) — reproduce & release weights**
- [ ] Finalize `effect_prompts.txt` (one `name: prompt` per effect, in the teacher-LoRA order) and release it.
- [ ] Release the encoded text-embedding bank produced by `encode_prompts.py`.
- [ ] Train the HyperNet on the 51 Remade-AI teacher LoRAs and **release the reproduced Stage-1 weights** (HF Hub).
