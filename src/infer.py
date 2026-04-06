"""Interactive masked-token inference for debugging the MLM model.

Example:
    uv run src/infer.py --checkpoint outputs/neobert-he-mlm/checkpoint-1000 --text "מ[MASK]של[MASK]"

If no --text is given, reads lines from stdin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_model(checkpoint: str) -> "ModernBertForMaskedLM":
    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    model.eval()
    return model


def run(model, tokenizer: PreTrainedTokenizerFast, text: str, device: torch.device) -> None:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    preds = out.logits.argmax(-1)
    # Replace [MASK] tokens with predictions
    mask_id = tokenizer.mask_token_id
    result_ids = input_ids.clone()
    result_ids[input_ids == mask_id] = preds[input_ids == mask_id]

    from decode import ids_to_str
    print(f"input : {text}")
    print(f"output: {ids_to_str(result_ids[0].tolist(), tokenizer)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)

    if args.text:
        run(model, tokenizer, args.text, device)
    else:
        print("Enter masked Hebrew text (Ctrl+C to quit):")
        while True:
            try:
                text = input("> ")
                if text.strip():
                    run(model, tokenizer, text, device)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
