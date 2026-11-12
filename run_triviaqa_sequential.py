import subprocess
import time
import os
import sys
import json
import numpy as np
import pandas as pd

PYTHON_BIN = "/home/krudra_mtp/miniconda3/envs/tsv/bin/python"
SEEDS = [100, 2024, 7, 99, 42]

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print(f" Starting TriviaQA Rerun: Seeds {SEEDS} (Sequential, Fast Eval)")
print("=" * 60)

for s in SEEDS:
    ckpt_path = f"checkpoints/tsv_triviaqa_seed_{s}.pt"
    # Check if already done in all_experiments.jsonl with fixed code
    print(f"\n--- Running TriviaQA Seed {s} ---")
    t0 = time.time()
    cmd = [
        PYTHON_BIN, "tsv_experiment.py",
        "--experiment_id", "E2_E3_E4_seed_variance",
        "--dataset_name", "triviaqa",
        "--seed", str(s),
        "--component", "res",
        "--fast_eval", "1",
        "--save_checkpoint", ckpt_path
    ]
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[Error] Seed {s} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"Seed {s} finished in {elapsed:.1f}s, checkpoint saved to {ckpt_path}")

print("\n" + "=" * 60)
print(" All 5 TriviaQA Seeds Completed! Summary Table:")
print("=" * 60)

# Extract TriviaQA records from all_experiments.jsonl
recs = []
with open("results/all_experiments.jsonl") as f:
    for line in f:
        if line.strip():
            d = json.loads(line.strip())
            if d.get("dataset") == "triviaqa" and d.get("experiment_id") == "E2_E3_E4_seed_variance":
                recs.append(d)

df = pd.DataFrame(recs)
# Deduplicate by seed (keep latest)
df = df.drop_duplicates(subset=['seed'], keep='last').sort_values('seed')

print(df[['seed', 'phase1_val_selected_test_auroc', 'val_selected_test_auroc', 'last_epoch_test_auroc', 'phase2_gain_honest']].to_string(index=False))

print("\nSample Std (n-1) Summary:")
for col in ['phase1_val_selected_test_auroc', 'val_selected_test_auroc', 'last_epoch_test_auroc', 'phase2_gain_honest']:
    vals = df[col].values
    m = np.mean(vals)
    s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    med = np.median(vals)
    print(f"{col}: {m:.4f} ± {s:.4f} (median: {med:.4f})")

