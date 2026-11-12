import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
import torch.nn.functional as F

from cross_eval import load_eval_data, test_model, HF_NAMES, ALL_DATASETS
from llm_layers import add_tsv_layers

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_NAME = "llama3.1-8B"
EVAL_DATASETS = ['tqa', 'triviaqa', 'sciq', 'nq_open', 'hotpotqa']
SEEDS = [42, 100, 2024, 7, 99]
BATCH_SIZE = 16

class ArgsShim:
    def __init__(self, model_name='llama3.1-8B', wild_ratio=0.75, str_layer=9, component='res', lam=5.0):
        self.model_name = model_name
        self.wild_ratio = wild_ratio
        self.str_layer = str_layer
        self.component = component
        self.lam = lam

args_shim = ArgsShim()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name_or_path = HF_NAMES[MODEL_NAME]

    print("=" * 70)
    print(" E1 CROSS-DATASET TRANSFER EVALUATION")
    print(f" Model: {model_name_or_path}")
    print(f" Targets: {EVAL_DATASETS}")
    print("=" * 70)

    # 1. Load Tokenizer & Base Model (Loaded ONCE)
    print("\n[1/3] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject TSV ParameterList
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    tsv = nn.ParameterList([
        nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        for _ in range(num_layers)
    ]).to(device)
    add_tsv_layers(model, tsv, [args_shim.lam], args_shim)

    # 3. Pre-load and cache all 5 target test sets (stored on CPU to keep GPU memory clean)
    print("\n[2/3] Pre-loading and caching test sets...")
    cache_path = "save_for_eval/all_test_splits_cpu_cached.pt"
    if os.path.exists(cache_path):
        print(f"Loading cached test sets from {cache_path}...")
        test_data = torch.load(cache_path)
    else:
        test_data = {}
        for ds in EVAL_DATASETS:
            print(f"  Caching {ds} test split...")
            prompts, labels = load_eval_data(ds, tokenizer, args_shim)
            test_data[ds] = ([p.cpu() for p in prompts], torch.tensor(labels))
            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos
            print(f"  -> {ds}: {len(labels)} samples (Truthful={n_pos}, Hallucinated={n_neg})")
        torch.save(test_data, cache_path)
        print(f"Saved cached test splits to {cache_path}")

    # 4. Evaluate Checkpoints
    print("\n[3/3] Evaluating checkpoints...")
    sources = ['tqa', 'triviaqa']
    results_matrix = {src: {tgt: [] for tgt in EVAL_DATASETS} for src in sources}
    detailed_results = []

    for src in sources:
        print(f"\n--- Source Dataset: {src} ---")
        for s in SEEDS:
            ckpt_path = f"checkpoints/tsv_{src}_seed_{s}.pt"
            if not os.path.exists(ckpt_path):
                print(f"[Warning] Checkpoint {ckpt_path} not found! Skipping.")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            tsv.load_state_dict(ckpt['tsv'])
            centroids = ckpt['centroids'].to(device)

            print(f"Evaluating Source: {src} | Seed: {s} | Checkpoint: {ckpt_path}")
            row_res = {'source': src, 'seed': s}

            for tgt in EVAL_DATASETS:
                prompts, labels = test_data[tgt]
                predictions, gt_labels = test_model(
                    model, centroids, prompts, labels, device, BATCH_SIZE
                )
                raw_auroc = roc_auc_score(gt_labels.numpy(), predictions.numpy())
                effective_auroc = max(raw_auroc, 1.0 - raw_auroc)
                results_matrix[src][tgt].append(effective_auroc)
                row_res[tgt] = effective_auroc
                torch.cuda.empty_cache()
                print(f"  -> Target {tgt:<10}: Effective AUROC = {effective_auroc:.4f} (raw: {raw_auroc:.4f})")

            detailed_results.append(row_res)

    # Save detailed results
    os.makedirs("results", exist_ok=True)
    with open("results/e1_transfer_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    # 5. Build Summary Transfer Matrix Table
    print("\n" + "=" * 80)
    print(" E1 CROSS-DATASET TRANSFER MATRIX (Mean ± Sample Std [ddof=1])")
    print("=" * 80)

    summary_rows = []
    for src in sources:
        row = {'Source Dataset': 'TruthfulQA' if src == 'tqa' else 'TriviaQA'}
        for tgt in EVAL_DATASETS:
            scores = results_matrix[src][tgt]
            if len(scores) > 0:
                mean_val = np.mean(scores)
                std_val = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                row[tgt] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[tgt] = "N/A"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    df_summary.to_csv("results/e1_transfer_matrix.csv", index=False)
    print("\nTransfer matrix saved to results/e1_transfer_matrix.csv")

if __name__ == '__main__':
    main()

