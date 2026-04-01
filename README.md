# Anonymous Submission: CoreVQA-Benchmark

> This repository is prepared for **double-blind peer review**.
> Please do not attempt to deanonymize the authors from this material.

## 1. What this repository contains

This repository provides the resources needed to evaluate Large Multimodal Models (LMMs) on a VQA benchmark designed around **total visual occlusion** settings.

Included materials (current release):
- `README.md` — instructions for reviewers.
- `evaluate.py` — evaluation script (if included in the submission package).
- `anonymous_data.jsonl` — anonymized benchmark annotations (if included in the submission package).

> Note: Some files may be omitted in the public/archival preview depending on venue policy and supplementary limits.

## 2. Anonymity statement (for reviewers)

To comply with double-blind requirements:
- No author names, affiliations, acknowledgements, or project identifiers are provided.
- Dataset fields that could reveal identity are masked or removed.
- Repository text avoids links to personal/institutional pages.

If you observe any potentially identifying artifact, please treat it as accidental metadata and ignore it for review purposes.

## 3. Benchmark summary

The benchmark studies LMM reasoning behavior under restricted visual input.

- **Task type**: Multiple-choice Visual Question Answering (VQA).
- **Focus**: Reasoning robustness and consistency under occlusion-like conditions.
- **Primary outputs**: accuracy-style metrics and per-sample prediction logs.

## 4. Expected data format

Each sample in `anonymous_data.jsonl` is a JSON object (one sample per line) with fields conceptually similar to:

```json
{
  "id": "sample_000001",
  "question": "...",
  "choices": ["A", "B", "C", "D"],
  "answer": "B",
  "meta": {
    "split": "test",
    "category": "occlusion_reasoning"
  }
}
```

The exact schema may vary slightly across supplementary versions, but core fields (`id`, `question`, `choices`, `answer`) are preserved.

## 5. Quick start

### 5.1 Environment

Recommended:
- Python 3.9+
- `pip` or `conda`

Install dependencies (example):

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided in the supplementary package, install minimal dependencies manually (commonly `numpy`, `pandas`, `tqdm`).

### 5.2 Run evaluation

Example command:

```bash
python evaluate.py \
  --data_path anonymous_data.jsonl \
  --pred_path <your_predictions.jsonl> \
  --output_dir results/
```

### 5.3 Prediction format

Your prediction file should typically contain one JSON object per line with:

```json
{
  "id": "sample_000001",
  "prediction": "B"
}
```

## 6. Reproducibility notes

For fair comparison, we recommend:
- Use deterministic decoding when possible.
- Report model name/version, prompt template, and inference hyperparameters.
- Run on the full evaluation split without manual filtering.

## 7. What reviewers may want to report

Suggested reporting items:
- Overall accuracy (or official primary metric).
- Accuracy by category/split (if metadata is available).
- Error patterns under severe occlusion conditions.

## 8. Limitations and scope

- This benchmark evaluates a specific reasoning setting and should not be interpreted as general intelligence ranking.
- Performance may be sensitive to prompting and model API versions.
- Some qualitative analysis assets may be excluded from the anonymous package to avoid deanonymization risk.

## 9. Post-review update plan

After the review process, the camera-ready version will include:
- Full documentation and complete dependency lockfile.
- Extended analysis scripts and optional visualization utilities.
- Public issue tracker and long-term maintenance details.

---

If any command/file path in this anonymous package differs from your local setup, please follow the intended interface and adapt paths accordingly.
