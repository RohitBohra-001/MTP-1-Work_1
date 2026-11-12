import subprocess
import time
import os

PYTHON_BIN = "/home/krudra_mtp/miniconda3/envs/tsv/bin/python"
SEEDS = [42, 100, 2024, 7, 99]

os.makedirs("checkpoints", exist_ok=True)

for s in SEEDS:
    ckpt_path = f"checkpoints/tsv_tqa_seed_{s}.pt"
    if os.path.exists(ckpt_path):
        print(f"TruthfulQA checkpoint {ckpt_path} already exists. Skipping.")
        continue
    print(f"Generating TruthfulQA Seed {s} checkpoint...")
    cmd = [
        PYTHON_BIN, "tsv_experiment.py",
        "--experiment_id", "E2_tqa_save_ckpt",
        "--dataset_name", "tqa",
        "--seed", str(s),
        "--component", "res",
        "--fast_eval", "1",
        "--save_checkpoint", ckpt_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved {ckpt_path}")

