# CoreVQA-Benchmark

Evaluation code for large vision-language models on the CoreVQA-Benchmark dataset.

## Setup

```bash
conda create -n corevqa python=3.10 -y
conda activate corevqa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pillow tqdm
```

## Data

Download the dataset from [Baidu Netdisk](TODO) and extract it into the `data/` directory:

```
data/
├── images_mask/
├── images_obb/
└── corevqa_benchmark.jsonl
```

## Evaluation

```bash
python eval.py
```

Other options:

```bash
python eval.py \
    --model <model_name_or_path> \
    --benchmark data/corevqa_benchmark.jsonl \
    --image_dir data/images_mask \
    --output results/output.jsonl \
    --device cuda
```

## Results

Results are saved to `results/` as JSONL, one record per sample:

```json
{
  "question_id": "...",
  "prediction": "B",
  "gt": "B",
  "score": 1.0,
  "combination": "V+P",
  "level": "Level2"
}
```

A summary of accuracy broken down by `combination` and `level` is printed at the end.

## Project Structure

```
.
├── data/                  # Dataset (download and extract here)
├── results/               # Evaluation outputs
├── eval.py                # Main evaluation script
└── README.md
```

## Citation

TODO
