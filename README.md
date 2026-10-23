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

## Results (latest run)

On simulated RAVEN-style puzzles (simulation only; 2,000 held-out test puzzles, 8-way answer accuracy), a standard MLP candidate scorer standing in for your CPCNet reached 15%, while a LightGBM ranker on rule-consistency features reached 100%, training about 13x faster.

| Metric | Value |
|---|---|
| baseline_score | 0.1503 |
| improved_score | 1 |
| baseline_score_std | 0.003793 |
| improved_score_std | 0 |
| improvement | 0.8497 |
| chance_accuracy | 0.125 |
| baseline_cv_accuracy | 0.147 |
| improved_cv_accuracy | 0.9977 |
| baseline_train_time_s | 4.861 |
| improved_train_time_s | 0.3824 |
| baseline_model_bytes | 1236615 |
| improved_model_bytes | 189069 |
| train_speedup_x | 12.71 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulation only: results do not transfer to real RAVEN/I-RAVEN/PGM or to CPCNet. The improved model uses hand-coded knowledge of the rule families (strong inductive bias), whereas the MLP must learn them from raw one-hot input. Rule-family overlap in random rows can make a few puzzles ambiguous. Test split is 33% (4000/2000 as specified, above the usual 20-30%). Train time of the improved model includes feature extraction; model size is the pickled size in bytes. Single process, small CPU budget for both.

## Reference

This project is motivated by the following paper:

> Y Yang, D Sanyal, J Ainooson, J Michelson (2023). *A cognitively-inspired neural architecture for visual abstract reasoning using contrastive perceptual and conceptual processing*. arXiv preprint arXiv. https://arxiv.org/abs/2309.10532

Paper author: Prof. Maithilee Kunda, Vanderbilt University (mkunda@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
