# Rule-consistency boosting vs an MLP candidate scorer on simulated Raven-style puzzles

## Motivation
The paper "A cognitively-inspired neural architecture for visual abstract reasoning using contrastive perceptual and conceptual processing" (2023) introduces CPCNet, a deep network that reaches very high accuracy on RAVEN-style benchmarks with a weak inductive bias. Such networks are expensive to train. This experiment asks whether a small gradient-boosted model fed with explicit rule-consistency features can pick the right answer at a much lower training cost.

## Question
Can a small LightGBM scorer on rule-consistency features beat a standard MLP candidate scorer (a CPU-only stand-in for a heavy CPCNet-style network) at choosing the correct answer panel, while training faster?

## Dataset
A seeded numpy simulation of RAVEN-style 3x3 puzzles (3 attributes with 10 values each; row rules: constant, progression, arithmetic, distribute-three). The 8 candidates follow the I-RAVEN balanced-tree construction, so the answer cannot be guessed from the candidates alone. 4,000 puzzles are used for training and 2,000 for testing. This is **not** the paper's dataset, and nothing is downloaded.

## Models
- **Baseline:** sklearn `MLPClassifier` scoring a one-hot encoding of (8 context panels + candidate). This is not CPCNet.
- **Improved:** `LightGBM` binary scorer on per-attribute rule-consistency features computed from rows 1-2 and the candidate-completed row 3.

Both use the same data, split, and evaluation (highest-scoring candidate wins). Hyper-parameters are chosen by cross-validation on the training set only, and the test set is used once at the end. Each model is run with seeds 42, 43 and 44. Secondary metrics are training time and model size.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> Y Yang, D Sanyal, J Ainooson, J Michelson (2023). *A cognitively-inspired neural architecture for visual abstract reasoning using contrastive perceptual and conceptual processing*. arXiv preprint arXiv. https://arxiv.org/abs/2309.10532

Paper author: Prof. Maithilee Kunda, Vanderbilt University (mkunda@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
