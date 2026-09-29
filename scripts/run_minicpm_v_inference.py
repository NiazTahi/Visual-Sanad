import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


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


def stringify_answer(answer):
    if isinstance(answer, str):
        return answer
    chunks = []
    for chunk in answer:
        chunks.append(str(chunk))
    return "".join(chunks)


def generate_one(model, tokenizer, prompt, image_path, max_new_tokens):
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, prompt]}]
    kwargs = {
        "msgs": msgs,
        "tokenizer": tokenizer,
        "enable_thinking": False,
        "stream": False,
        "max_new_tokens": max_new_tokens,
    }
    try:
        answer = model.chat(**kwargs)
    except TypeError:
        kwargs.pop("max_new_tokens")
        answer = model.chat(**kwargs)
    return stringify_answer(answer).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4_5")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
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

    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    image_root = Path(args.image_root)
    for index, trial in enumerate(pending, start=1):
        started = time.time()
        try:
            raw = generate_one(
                model,
                tokenizer,
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
            "displayed_reference_model": payload.get("display_reference")
            or payload.get("displayed_reference")
            or payload.get("cited_reference", ""),
            "confidence_model": payload.get("confidence", ""),
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
