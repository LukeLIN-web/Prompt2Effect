import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from transformers import T5Tokenizer, T5EncoderModel

"""
python encode_prompts.py \
  --effect-prompts /path/to/effect_prompts.txt \
  --language-model-ckpt /path/to/text_encoder_ckpt.pt \
  --output /path/to/effects_text_bank.pt
"""

def _parse_effect_prompts_txt(path: str) -> Tuple[List[str], List[str]]:
    """
    Parse effect_prompts.txt. We treat lines of the form `name: text...` as a new item
    and accumulate any subsequent non-empty, non-new-key lines into that same item's text.
    Items are kept in file order. Unlabeled stray lines before the first key are ignored.
    """
    names: List[str] = []
    texts: List[str] = []
    cur_name = None
    cur_lines: List[str] = []
    # Added \- inside the brackets to capture names with hyphens
    key_pat = re.compile(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = key_pat.match(line)
            if m:
                # flush previous
                if cur_name is not None:
                    names.append(cur_name)
                    texts.append(" ".join(cur_lines).strip())
                cur_name = m.group(1)
                first_text = m.group(2).strip()
                cur_lines = [first_text] if first_text else []
            else:
                if cur_name is not None:
                    cur_lines.append(line)
                # else: ignore stray text before first key

    if cur_name is not None:
        names.append(cur_name)
        texts.append(" ".join(cur_lines).strip())

    return names, texts


def encode_effect_prompts_to_bank(
    effect_prompts_txt: str,
    language_model_ckpt: str,
    device: str = "cuda",
    max_length: int = 512,
    torch_dtype = torch.float32,
    compile_lm: bool = False,
    batch_size: int = 16,
) -> Dict[str, object]:
    """
    Build a bank of embeddings aligned to effect_prompts.txt order.
    Returns a dict with:
      - names:   [L] keys from the file
      - prompts: [L] the full prompts
      - embeddings: torch.Tensor [L, T, E] (float32)
      - masks:      torch.Tensor [L, T] (bool) -> True where padding
    """
    
    names, prompts = _parse_effect_prompts_txt(effect_prompts_txt)
    if len(names) == 0:
        raise ValueError(f"No 'name: text' items found in {effect_prompts_txt}")

    print(f"[encode] Loading T5 from {language_model_ckpt} ...")
    
    # Load HuggingFace T5 directly to bypass the broken wrapper
    # legacy=False is recommended to avoid the warning you saw earlier
    tokenizer = T5Tokenizer.from_pretrained(language_model_ckpt, legacy=False)
    model = T5EncoderModel.from_pretrained(language_model_ckpt)
    model.to(device)
    
    if compile_lm:
        model = torch.compile(model, mode="max-autotune")

    all_embeds: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = [] 

    print(f"[encode] Processing {len(prompts)} prompts in batches of {batch_size}...")

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        
        # Tokenize the batch
        encoded = tokenizer(
            chunk, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=max_length,
            add_special_tokens=True
        )
        
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeds = outputs.last_hidden_state  # [B, 512, 4096]
            embeds = embeds.to(torch_dtype)

        # Generate Mask: PyTorch convention is often True = Padding (Ignore), False = Content (Keep)
        # We manually calculate where the text ends (EOT) to set the mask.
        batch_mask = torch.ones((embeds.shape[0], max_length), dtype=torch.bool, device=embeds.device)

        # Find EOT (End of Text) location manually
        for b in range(input_ids.shape[0]):
            # T5 uses 1 (</s>) as the EOS token.
            # Find the first index where the token is the EOS token
            eos_indices = (input_ids[b] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            
            if len(eos_indices) > 0:
                eot_loc = eos_indices[0].item() # Location of first </s>
                # Keep content up to (and including) EOS token if desired, 
                # or just up to EOS. Standard is often to include EOS.
                # Here we set False (Keep) for indices 0 to eot_loc
                batch_mask[b, :eot_loc + 1] = False 
            else:
                # Truncated or no EOS found, keep full length
                batch_mask[b, :] = False

        all_embeds.append(embeds.cpu())
        all_masks.append(batch_mask.cpu())

    bank = {
        "names": names,
        "prompts": prompts,
        "embeddings": torch.cat(all_embeds, dim=0).contiguous(),  # [L,T,E]
        # "masks": torch.cat(all_masks, dim=0).contiguous(),        # [L,T]
    }
    return bank


def _cli_main():
    p = argparse.ArgumentParser("Encode effect_prompts.txt into a bank of real text embeddings")
    p.add_argument("--effect-prompts", type=str, required=True, help="Path to effect_prompts.txt")
    p.add_argument("--language-model-ckpt", type=str, required=True, help="LLM/TextEncoder checkpoint")
    p.add_argument("--output", type=str, required=True, help="Output .pt file")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=512, help="Token length; MUST match train --text-len")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--compile", action="store_true", help="torch.compile the LM")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Use bfloat16 as requested in your previous snippet
    bank = encode_effect_prompts_to_bank(
        effect_prompts_txt=args.effect_prompts,
        language_model_ckpt=args.language_model_ckpt,
        device=args.device,
        max_length=args.max_length,
        compile_lm=args.compile,
        batch_size=args.batch_size,
        torch_dtype=torch.bfloat16
    )
    torch.save(bank, args.output)
    print(f"[encode_prompts] wrote embeddings for {len(bank['prompts'])} prompts to {args.output}; "
          f"shape={tuple(bank['embeddings'].shape)} (L,T,E)")


if __name__ == "__main__":
    _cli_main()