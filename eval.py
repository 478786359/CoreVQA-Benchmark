"""
eval.py
-------
Evaluate a vision-language model on CoreVQA-Benchmark.
Each sample is evaluated twice: once with the OBB image and once with the mask image.

Usage:
    python eval.py
    python eval.py --model <model_name_or_path> --output results/output.jsonl
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_question(sample: dict) -> str:
    options = sample["options"]
    opts_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{sample['question']}\n{opts_str}\n\nAnswer with the option letter directly."


def extract_answer(text: str) -> str:
    for pat in [
        r"(?:answer is|answer:)\s*([A-D])\b",
        r"^([A-D])\s*[.)]\s",
        r"\b([A-D])\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return ""


def run_inference(model, processor, image_path: str, question: str, device: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def print_summary(results: list):
    for prompt_type in ["obb", "mask"]:
        subset = [r for r in results if r["prompt_type"] == prompt_type]
        if not subset:
            continue
        acc = sum(r["score"] for r in subset) / len(subset) * 100
        print(f"\n[{prompt_type.upper()}] Overall accuracy: {acc:.1f}% ({len(subset)} samples)")

        combo = defaultdict(list)
        for r in subset:
            combo[r["combination"]].append(r["score"])
        print(f"  {'combination':<16} {'acc%':>6} {'n':>6}")
        for k, v in sorted(combo.items()):
            print(f"  {k:<16} {sum(v)/len(v)*100:>5.1f}% {len(v):>6}")

        level = defaultdict(list)
        for r in subset:
            level[r["level"]].append(r["score"])
        print(f"\n  {'level':<16} {'acc%':>6} {'n':>6}")
        for k, v in sorted(level.items()):
            print(f"  {k:<16} {sum(v)/len(v)*100:>5.1f}% {len(v):>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--benchmark", default="data/corevqa_benchmark.jsonl")
    parser.add_argument("--data_dir", default="data",
                        help="Base directory for image paths in benchmark")
    parser.add_argument("--output", default="results/output.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    samples = load_jsonl(args.benchmark)
    print(f"Loaded {len(samples)} samples, running 2 passes each ({len(samples)*2} total)\n")

    results = []
    with open(args.output, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="Evaluating"):
            question = build_question(sample)
            gt = sample["answer"]
            meta = sample.get("meta", {})

            for prompt_type, img_key in [("obb", "img_obb"), ("mask", "img_mask")]:
                img_path = str(Path(args.data_dir) / sample[img_key])
                if not Path(img_path).exists():
                    tqdm.write(f"⚠ Missing: {img_path}")
                    continue

                raw_output = run_inference(model, processor, img_path, question, args.device)
                prediction = extract_answer(raw_output)
                score = 1.0 if prediction == gt else 0.0

                record = {
                    "question_id": sample["question_id"],
                    "prompt_type": prompt_type,
                    "prediction": prediction,
                    "gt": gt,
                    "score": score,
                    "combination": meta.get("combination", ""),
                    "level": meta.get("level", ""),
                    "raw_output": raw_output,
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_summary(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
