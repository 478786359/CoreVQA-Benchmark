# CoreVQA-Benchmark
CoRe-VQA: Probing Systematic Reasoning Deficits in LMMs under Total Occlusion

This repository contains the dataset and evaluation toolkit for the paper:  
CoRe-VQA: Probing Systematic Reasoning Deficits in LMMs under Total Occlusion Submitted for double-blind review.

## 1. Overview
This benchmark is designed to evaluate the [e.g., reasoning consistency] of Large Multimodal Models (LMMs) under specific scenarios.
- **Total Samples**: [e.g., 10,000+]
- **Task**: Multiple-choice Visual Question Answering (VQA).
- **Domain**: [e.g., Object reasoning under visual occlusion].

## 2. Repository Structure
```text
.
├── README.md                # This file
├── evaluate.py              # Standard evaluation & data loading script
└── anonymous_data.jsonl     # Masked annotation file (strictly anonymized)
