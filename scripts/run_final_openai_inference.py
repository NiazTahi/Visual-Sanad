import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


DEFAULT_PROMPT_PATH = "prompts/visual_sanad_inference_prompt_v1.txt"


def load_api_key(path):
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"sk-[A-Za-z0-9_-]+", text)
    if not match:
        raise RuntimeError(f"No OpenAI key pattern found in {path}; or set OPENAI_API_KEY")
    return match.group(0)


def image_data_url(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
    rows = read_csv(path)
    return {row["trial_id"] for row in rows if not row.get("error")}


def parse_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def parse_label(text, payload):
    label = str(payload.get("label", "")).strip().upper()
    if label in {"VALID", "INVALID", "ABSTAIN"}:
        return label
    upper = text.upper()
    for candidate in ("INVALID", "VALID", "ABSTAIN"):
        if re.search(rf"\b{candidate}\b", upper):
            return candidate
    return "UNPARSED"


def call_model(client, model, prompt, image_path, max_output_tokens, temperature):
    kwargs = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url(image_path)},
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.responses.create(**kwargs)
    return response.output_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="final_rendered_locked_2400/manifest.csv")
    parser.add_argument("--image-root", default="final_rendered_locked_2400")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--key-file", default="openai_keys.txt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    client = OpenAI(api_key=load_api_key(args.key_file))
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    manifest = read_csv(args.manifest)
    done = completed_trial_ids(args.out)
    pending = [row for row in manifest if row["trial_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    image_root = Path(args.image_root)
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "manifest": args.manifest,
                "image_root": args.image_root,
                "prompt_file": args.prompt_file,
                "prompt": prompt,
                "max_output_tokens": args.max_output_tokens,
                "temperature": args.temperature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(pending)
    for index, trial in enumerate(pending, start=1):
        image_path = image_root / trial["image_path"]
        started = time.time()
        try:
            raw = call_model(
                client,
                args.model,
                prompt,
                image_path,
                args.max_output_tokens,
                args.temperature,
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
            "model": args.model,
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
                    "total": total,
                    "trial_id": trial["trial_id"],
                    "model": args.model,
                    "prediction": prediction,
                    "gold": trial["gold_label"],
                    "error": bool(error),
                }
            ),
            flush=True,
        )
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
