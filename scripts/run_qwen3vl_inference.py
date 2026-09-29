import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_trial_ids(path):
    if not Path(path).exists():
        return set()
    return {row["trial_id"] for row in read_csv(path) if not row.get("error")}


def parse_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def parse_label(text, payload):
    label = str(payload.get("label") or payload.get("verdict") or "").strip().upper()
    if label in {"VALID", "INVALID", "ABSTAIN"}:
        return label
    upper = text.upper()
    for candidate in ("INVALID", "VALID", "ABSTAIN"):
        if re.search(rf"\b{candidate}\b", upper):
            return candidate
    return "UNPARSED"


def generate_one(model, processor, prompt, image_path, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt-file", default="prompts/visual_sanad_inference_prompt_natural_v1.txt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    rows = read_csv(args.manifest)
    done = completed_trial_ids(args.out)
    pending = [row for row in rows if row["trial_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    image_root = Path(args.image_root)
    for index, trial in enumerate(pending, start=1):
        started = time.time()
        try:
            raw = generate_one(
                model,
                processor,
                prompt,
                image_root / trial["image_path"],
                args.max_new_tokens,
            )
            payload = parse_json_object(raw)
            error = ""
        except Exception as exc:
            raw = ""
            payload = {}
            error = f"{type(exc).__name__}: {exc}"

        prediction = parse_label(raw, payload)
        result = {
            **trial,
            "model": args.model_id,
            "prediction": prediction,
            "correct": str(prediction == trial["gold_label"]),
            "displayed_reference_model": str(payload.get("displayed_reference", "")),
            "confidence_model": str(payload.get("confidence", "")),
            "raw_output": raw,
            "error": error,
            "latency_s": f"{time.time() - started:.3f}",
        }
        append_csv(args.out, result)
        print(
            json.dumps(
                {
                    "n": index,
                    "total": len(pending),
                    "trial_id": trial["trial_id"],
                    "prediction": prediction,
                    "gold": trial["gold_label"],
                    "error": bool(error),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
