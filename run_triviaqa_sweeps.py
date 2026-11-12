import os
os.environ["PYTORCH_NO_CUDA_NVML"] = "1"
import subprocess
import time
import json
import os
import pandas as pd

PYTHON_BIN = "/home/krudra_mtp/miniconda3/envs/tsv/bin/python"

def is_already_done(exp_id, dataset, seed, layer, n_ex):
    jsonl_path = "./results/all_experiments.jsonl"
    if not os.path.exists(jsonl_path):
        return False
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line.strip())
            if (r.get("experiment_id") == exp_id and
                r.get("dataset") == dataset and
                r.get("seed") == seed and
                r.get("str_layer") == layer and
                r.get("num_exemplars") == n_ex):
                return True
    return False

def rebuild_csv():
    try:
        recs = []
        with open('./results/all_experiments.jsonl') as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line.strip()))
        pd.DataFrame(recs).to_csv('./results/summary_table.csv', index=False)
    except Exception as e:
        print(f"Warning rebuilding CSV: {e}")

def run_layer_sweep():
    print("\n=======================================================")
    print(" SUITE E5: Steering Layer Sweep on TriviaQA (3 Seeds/Layer)")
    print("=======================================================")
    layers = [2, 4, 6, 8, 9, 10, 12, 16, 20, 24, 28]
    seeds = [42, 100, 2024]
    
    for lyr in layers:
        for s in seeds:
            if is_already_done("E5_layer_sweep", "triviaqa", s, lyr, 32):
                print(f"Skipping Layer {lyr}, Seed {s} (already done)")
                continue

            # If layer 9, check if baseline E2_E3_E4 was done and copy it
            if lyr == 9 and is_already_done("E2_E3_E4_seed_variance", "triviaqa", s, 9, 32):
                print(f"Layer 9, Seed {s} already computed in baseline. Recording copy...")
                latest_match = None
                with open("./results/all_experiments.jsonl") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line.strip())
                            if r.get("experiment_id") == "E2_E3_E4_seed_variance" and r.get("dataset") == "triviaqa" and r.get("seed") == s:
                                latest_match = r
                if latest_match:
                    r_copy = dict(latest_match)
                    r_copy["experiment_id"] = "E5_layer_sweep"
                    with open("./results/all_experiments.jsonl", "a") as f_out:
                        f_out.write(json.dumps(r_copy) + "\n")
                rebuild_csv()
                continue

            print(f"\n--- Running Layer {lyr}, Seed {s} ---")
            t0 = time.time()
            cmd = [
                PYTHON_BIN, "tsv_experiment.py",
                "--experiment_id", "E5_layer_sweep",
                "--dataset_name", "triviaqa",
                "--seed", str(s),
                "--str_layer", str(lyr),
                "--lam", "5.0",
                "--num_exemplars", "32",
                "--num_selected_data", "128",
                "--batch_size", "16",
                "--fast_eval", "1",
                "--init_num_epochs", "20",
                "--aug_num_epochs", "20",
            ]
            res = subprocess.run(cmd)
            elapsed = time.time() - t0
            print(f"Layer {lyr}, Seed {s} finished in {elapsed:.1f}s (code: {res.returncode})")
            rebuild_csv()

def run_exemplar_sweep():
    print("\n=======================================================")
    print(" SUITE E6: Exemplar Budget Sweep on TriviaQA (3 Seeds/N)")
    print("=======================================================")
    budgets = [8, 16, 32, 64, 128]
    seeds = [42, 100, 2024]

    for n in budgets:
        for s in seeds:
            if is_already_done("E6_exemplar_budget", "triviaqa", s, 9, n):
                print(f"Skipping N={n}, Seed {s} (already done)")
                continue

            if n == 32 and is_already_done("E2_E3_E4_seed_variance", "triviaqa", s, 9, 32):
                print(f"N=32, Seed {s} already computed in baseline. Recording copy...")
                latest_match = None
                with open("./results/all_experiments.jsonl") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line.strip())
                            if r.get("experiment_id") == "E2_E3_E4_seed_variance" and r.get("dataset") == "triviaqa" and r.get("seed") == s:
                                latest_match = r
                if latest_match:
                    r_copy = dict(latest_match)
                    r_copy["experiment_id"] = "E6_exemplar_budget"
                    with open("./results/all_experiments.jsonl", "a") as f_out:
                        f_out.write(json.dumps(r_copy) + "\n")
                rebuild_csv()
                continue

            print(f"\n--- Running N={n}, Seed {s} ---")
            t0 = time.time()
            cmd = [
                PYTHON_BIN, "tsv_experiment.py",
                "--experiment_id", "E6_exemplar_budget",
                "--dataset_name", "triviaqa",
                "--seed", str(s),
                "--str_layer", "9",
                "--lam", "5.0",
                "--num_exemplars", str(n),
                "--num_selected_data", "128",
                "--batch_size", "16",
                "--fast_eval", "1",
                "--init_num_epochs", "20",
                "--aug_num_epochs", "20",
            ]
            res = subprocess.run(cmd)
            elapsed = time.time() - t0
            print(f"N={n}, Seed {s} finished in {elapsed:.1f}s (code: {res.returncode})")
            rebuild_csv()

if __name__ == "__main__":
    run_layer_sweep()
    run_exemplar_sweep()
    print("\nAll TriviaQA sweeps completed!")

