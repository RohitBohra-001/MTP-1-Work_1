import os
import sys
import subprocess
import argparse
import time

PYTHON_BIN = "/home/krudra_mtp/miniconda3/envs/tsv/bin/python"
SCRIPT = "tsv_experiment.py"

def run_cmd(cmd_args):
    cmd = [PYTHON_BIN, SCRIPT] + cmd_args
    print(f"\n>>> Running: {' '.join(cmd)}")
    start_time = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_time
    if res.returncode != 0:
        print(f"FAILED with returncode {res.returncode} (took {elapsed:.1f}s)")
        return False
    print(f"SUCCESS (took {elapsed:.1f}s)")
    return True

def run_suite_e2_e4_tqa():
    print("\n==================================================")
    print(" SUITE: E2, E3, E4 on TruthfulQA (5 Seeds)")
    print("==================================================")
    seeds = [42, 100, 2024, 7, 99]
    for s in seeds:
        run_cmd([
            "--experiment_id", "E2_E3_E4_seed_variance",
            "--dataset_name", "tqa",
            "--seed", str(s),
            "--str_layer", "9",
            "--lam", "5.0",
            "--num_exemplars", "32",
            "--num_selected_data", "128",
            "--batch_size", "16",
            "--init_num_epochs", "20",
            "--aug_num_epochs", "20",
        ])

def run_suite_e2_e4_triviaqa():
    print("\n==================================================")
    print(" SUITE: E2, E3, E4 on TriviaQA (5 Seeds)")
    print("==================================================")
    seeds = [42, 100, 2024, 7, 99]
    for s in seeds:
        run_cmd([
            "--experiment_id", "E2_E3_E4_seed_variance",
            "--dataset_name", "triviaqa",
            "--seed", str(s),
            "--str_layer", "9",
            "--lam", "5.0",
            "--num_exemplars", "32",
            "--num_selected_data", "128",
            "--batch_size", "16",
            "--init_num_epochs", "20",
            "--aug_num_epochs", "20",
        ])

def run_suite_e5_layer_sweep():
    print("\n==================================================")
    print(" SUITE: E5 Steering Layer Sweep on TQA (3 Seeds/Layer)")
    print("==================================================")
    layers = [2, 4, 6, 8, 9, 10, 12, 16, 20, 24, 28]
    seeds = [42, 100, 2024]
    for lyr in layers:
        for s in seeds:
            run_cmd([
                "--experiment_id", "E5_layer_sweep",
                "--dataset_name", "tqa",
                "--seed", str(s),
                "--str_layer", str(lyr),
                "--lam", "5.0",
                "--num_exemplars", "32",
                "--num_selected_data", "128",
                "--batch_size", "16",
                "--init_num_epochs", "20",
                "--aug_num_epochs", "20",
            ])

def run_suite_e6_exemplar_sweep():
    print("\n==================================================")
    print(" SUITE: E6 Exemplar Budget Sweep on TQA (3 Seeds/Budget)")
    print("==================================================")
    budgets = [8, 16, 32, 64, 128]
    seeds = [42, 100, 2024]
    for n in budgets:
        for s in seeds:
            run_cmd([
                "--experiment_id", "E6_exemplar_budget",
                "--dataset_name", "tqa",
                "--seed", str(s),
                "--str_layer", "9",
                "--lam", "5.0",
                "--num_exemplars", str(n),
                "--num_selected_data", "128",
                "--batch_size", "16",
                "--init_num_epochs", "20",
                "--aug_num_epochs", "20",
            ])

def run_suite_priority_b():
    print("\n==================================================")
    print(" SUITE: Priority B (E7, E8, E9, E10)")
    print("==================================================")
    # E7: Lambda sweep
    lambdas = [0.1, 0.5, 1.0, 5.0, 10.0]
    for lam in lambdas:
        run_cmd([
            "--experiment_id", "E7_steering_strength",
            "--dataset_name", "tqa",
            "--seed", "42",
            "--lam", str(lam),
            "--batch_size", "16",
        ])

    # E8: OT vs Simple pseudo-labeling
    for method in ["sinkhorn", "simple"]:
        for s in [42, 100, 2024]:
            run_cmd([
                "--experiment_id", "E8_ot_vs_simple",
                "--dataset_name", "tqa",
                "--seed", str(s),
                "--pseudo_label_method", method,
                "--batch_size", "16",
            ])

    # E9: Selected sample count K
    for k in [32, 64, 128, 256]:
        run_cmd([
            "--experiment_id", "E9_sample_count_k",
            "--dataset_name", "tqa",
            "--seed", "42",
            "--num_selected_data", str(k),
            "--batch_size", "16",
        ])

    # E10: BLEURT threshold sensitivity
    for th in [0.4, 0.5, 0.6]:
        run_cmd([
            "--experiment_id", "E10_bleurt_threshold",
            "--dataset_name", "tqa",
            "--seed", "42",
            "--thres_gt", str(th),
            "--batch_size", "16",
        ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="e2_e4_tqa",
                        choices=["e2_e4_tqa", "e2_e4_triviaqa", "e5_layer_sweep", "e6_exemplar_sweep", "priority_a", "priority_b", "all"])
    args = parser.parse_args()

    if args.suite == "e2_e4_tqa":
        run_suite_e2_e4_tqa()
    elif args.suite == "e2_e4_triviaqa":
        run_suite_e2_e4_triviaqa()
    elif args.suite == "e5_layer_sweep":
        run_suite_e5_layer_sweep()
    elif args.suite == "e6_exemplar_sweep":
        run_suite_e6_exemplar_sweep()
    elif args.suite == "priority_a":
        run_suite_e2_e4_tqa()
        run_suite_e2_e4_triviaqa()
        run_suite_e5_layer_sweep()
        run_suite_e6_exemplar_sweep()
    elif args.suite == "priority_b":
        run_suite_priority_b()
    elif args.suite == "all":
        run_suite_e2_e4_tqa()
        run_suite_e2_e4_triviaqa()
        run_suite_e5_layer_sweep()
        run_suite_e6_exemplar_sweep()
        run_suite_priority_b()

if __name__ == "__main__":
    main()

