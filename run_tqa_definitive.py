import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd

SEEDS = [42, 100, 2024, 7, 99]
OUT_FILE = "results/tqa_definitive_5seeds.jsonl"
PYTHON_BIN = "/home/krudra_mtp/miniconda3/envs/tsv/bin/python"

os.makedirs("results", exist_ok=True)

# Check existing runs in OUT_FILE
completed_seeds = set()
if os.path.exists(OUT_FILE):
    with open(OUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    completed_seeds.add(data['seed'])
                except Exception:
                    pass

# Check if seed 42 was recorded in all_experiments.jsonl from our test run
if 42 not in completed_seeds and os.path.exists("results/all_experiments.jsonl"):
    with open("results/all_experiments.jsonl", 'r') as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line.strip())
                    if d.get('experiment_id') == 'E2_tqa_definitive_test42' and d.get('seed') == 42:
                        d['experiment_id'] = 'E2_tqa_definitive'
                        with open(OUT_FILE, 'a') as out_f:
                            out_f.write(json.dumps(d) + '\n')
                        completed_seeds.add(42)
                        print("[Info] Seed 42 result recovered from verified test run.")
                        break
                except Exception:
                    pass

for seed in SEEDS:
    if seed in completed_seeds:
        print(f"[Skip] Seed {seed} already completed in {OUT_FILE}.")
        continue
    
    print(f"\n{'='*60}")
    print(f" Running Definitive TruthfulQA: Seed {seed}")
    print(f"{'='*60}")
    
    cmd = [
        PYTHON_BIN, "tsv_experiment.py",
        "--dataset_name", "tqa",
        "--seed", str(seed),
        "--component", "res",
        "--fast_eval", "0",
        "--experiment_id", "E2_tqa_definitive"
    ]
    
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[Error] Seed {seed} failed with return code {res.returncode}")
        sys.exit(res.returncode)
        
    # Append the last line from all_experiments.jsonl to OUT_FILE
    with open("results/all_experiments.jsonl", 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        last_rec = json.loads(lines[-1])
        if last_rec.get('seed') == seed and last_rec.get('experiment_id') == 'E2_tqa_definitive':
            with open(OUT_FILE, 'a') as out_f:
                out_f.write(json.dumps(last_rec) + '\n')
            completed_seeds.add(seed)

print("\n" + "="*70)
print(" ALL 5 TRUTHFULQA SEEDS COMPLETED. SUMMARY TABLE:")
print("="*70)

records = []
with open(OUT_FILE, 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line.strip()))

df = pd.DataFrame(records)
print(df[['seed', 'phase1_val_selected_test_auroc', 'val_selected_test_auroc', 'test_snooped_best_auroc', 'last_epoch_test_auroc', 'phase1_last_epoch_test_auroc']])

