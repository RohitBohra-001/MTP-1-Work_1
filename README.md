# Truthfulness Separator Vector (TSV) — Reproducibility & Evaluation Study

This repository contains the codebase, splits, checkpoints, and empirical results for **Chapters 4 and 7** of the M.Tech thesis:
> **"Detecting Hallucinations from the Internal Representations of Large Language Models: A Study of Steering-Based and Spectral Detectors"**  
> Rohit Bohra (22EE3AI21) — Department of Artificial Intelligence, IIT Kharagpur  
> Supervised by Prof. Koustav Rudra

This work builds upon the official implementation of **TSV** ([Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917), ICML 2025) by Park et al. (`deeplearning-wisc/tsv`), and extends it to evaluate:
1. **RQ1 (Reproduction):** Single-run reproduction of the unmodified repository defaults.
2. **RQ2 (Seed Stability):** Multi-seed evaluation across 5 random seeds (`42, 100, 2024, 7, 99`) on TruthfulQA and TriviaQA.
3. **RQ3 (Checkpoint Selection & Test-Set Leakage):** Comparing honest 100-sample validation selection against test-set-selected checkpoints (as used in the released script) and unstopped last-epoch training.
4. **RQ4 (Phase Ablation):** Isolating Phase 1 (exemplar steering) vs. Phase 1+2 (Optimal-Transport pseudo-labeling + retraining) under honest validation selection.
5. **RQ5 (Cross-Dataset Transfer):** Evaluating all 10 validation-selected checkpoints across 5 QA benchmarks (`TruthfulQA`, `TriviaQA`, `SciQ`, `NQ-Open`, `HotpotQA`).

---

## Key Results (LLaMA-3.1-8B, Layer 9, $N=32$)

### Multi-Seed Performance & Checkpoint Selection (Mean ± Sample Std, $n-1$)

| Metric | TruthfulQA (5 Seeds) | TriviaQA (5 Seeds) |
| :--- | :---: | :---: |
| **Phase 1 Val-Selected AUROC** | $0.7502 \pm 0.0954$ | $0.7850 \pm 0.0476$ |
| **Full Pipeline Val-Selected** | **$0.7576 \pm 0.0801$** | **$0.8282 \pm 0.0107$** |
| **Full Pipeline Test-Snooped** | $0.7820 \pm 0.0770$ | — |
| **Last-Epoch AUROC (Epoch 40)** | $0.6677 \pm 0.0714$ | $0.7743 \pm 0.1084$ |
| **Honest Phase 2 Gain** | $+0.0074 \pm 0.0165$ | $+0.0433 \pm 0.0456$ |
| **Leakage Inflation (Test − Val)** | $+0.0245 \pm 0.0248$ | — |

### Cross-Dataset Transfer Matrix (Mean ± Sample Std over 5 Seeds)

| Train ↓ / Test → | TruthfulQA | TriviaQA | SciQ | NQ-Open | HotpotQA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TruthfulQA** | **$0.7576 \pm 0.0801$** | $0.6719 \pm 0.1193$ | $0.6739 \pm 0.1252$ | $0.6110 \pm 0.0307$ | $0.6405 \pm 0.1069$ |
| **TriviaQA** | $0.6997 \pm 0.0133$ | **$0.8282 \pm 0.0107$** | $0.7898 \pm 0.0125$ | $0.6563 \pm 0.0163$ | $0.7684 \pm 0.0127$ |

---

## Repository Structure

```text
.
├── tsv_main.py                   # Original TSV training & evaluation script
├── train_utils.py                # Sinkhorn pseudo-labeling, OT loss, EMA centroids
├── llm_layers.py                 # Residual stream steering hooks
├── sinkhorn_knopp.py             # Optimal transport Sinkhorn-Knopp solver
├── cache_utils.py                # Generation & activation caching utilities
├── cross_eval.py                 # Base cross-dataset evaluation helper
│
├── tsv_experiment.py             # Extended experiment runner (val/test/last-epoch tracking, Phase 1/2 ablation, checkpoint saving)
├── run_tqa_definitive.py         # 5-seed TruthfulQA evaluation runner (full per-epoch test & val logging)
├── run_tqa_checkpoints.py        # Saves validation-selected TruthfulQA checkpoints across 5 seeds
├── run_triviaqa_sequential.py    # 5-seed TriviaQA evaluation & checkpoint runner
├── evaluate_transfer_matrix.py   # E1 cross-dataset transfer matrix evaluator (all 10 checkpoints × 5 benchmarks)
├── run_experiments.py            # Orchestration utilities
├── rebuild_csv.py                # Aggregates JSONL results into summary CSV
│
├── data_indices/                 # Fixed train/test & exemplar indices (.npy)
├── ml_*_bleurt_score.npy         # Precomputed BLEURT ground-truth labels
├── checkpoints/                  # Validation-selected checkpoints for all 5 seeds (TruthfulQA & TriviaQA)
├── results/
│   ├── e1_transfer_matrix.csv    # Summary 2×5 cross-dataset transfer matrix
│   ├── e1_transfer_detailed.json # Per-seed cross-dataset AUROC scores
│   ├── tqa_definitive_5seeds.jsonl
│   ├── all_experiments.jsonl     # Full experiment logs
│   ├── summary_table.csv
│   └── curves/                   # Per-epoch validation and test AUROC trajectories
├── tsv.yml / requirements.txt    # Environment specification
└── *.sh                          # Generation, BLEURT labeling, and training shell scripts
```

---

## Setup & Reproduction

1. **Create the environment:**
   ```bash
   conda env create -f tsv.yml
   conda activate tsv
   ```

2. **Set your Hugging Face access token** (required for `meta-llama/Meta-Llama-3.1-8B`):
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

3. **Run the multi-seed experiments:**
   - **TruthfulQA (5 seeds, full per-epoch evaluation):**
     ```bash
     python run_tqa_definitive.py
     ```
   - **TriviaQA (5 seeds, validation-selected + checkpoint saving):**
     ```bash
     python run_triviaqa_sequential.py
     ```
   - **Cross-Dataset Transfer Matrix (E1):**
     ```bash
     python evaluate_transfer_matrix.py
     ```

---

## Acknowledgements & Citation

This codebase is built on top of the original TSV repository ([deeplearning-wisc/tsv](https://github.com/deeplearning-wisc/tsv)):

```bibtex
@inproceedings{park2025steer,
  title={Steer {LLM} Latents for Hallucination Detection},
  author={Seongheon Park and Xuefeng Du and Min-Hsuan Yeh and Haobo Wang and Yixuan Li},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

