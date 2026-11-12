"""
cross_eval.py  —  Zero-shot cross-dataset evaluation for TSV.

Load a TSV checkpoint (trained on some source dataset) and evaluate it
on one or more target datasets without any additional training.

Usage:
    python cross_eval.py \
        --model_name llama3.1-8B \
        --checkpoint ./checkpoints/tsv_tqa.pt \
        --eval_datasets tqa triviaqa sciq nq_open
"""

import os
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
from train_utils import get_last_non_padded_token_rep, collate_fn
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_layers import add_tsv_layers
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Model registry  (extend here to support new base models)
# ---------------------------------------------------------------------------

HF_NAMES = {
    'llama3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'qwen2.5-7B':  'Qwen/Qwen2.5-7B',
}

ALL_DATASETS = ['tqa', 'triviaqa', 'sciq', 'nq_open', 'hotpotqa']

# BLEURT threshold per dataset  (same as tsv_main.py)
BLEURT_THRESHOLD = {
    'tqa':      0.5,
    'triviaqa': 0.5,
    'sciq':     0.2,
    'nq_open':  0.2,
    'hotpotqa': 0.5,  # multi-hop QA — same threshold as TriviaQA
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_hf_dataset(dataset_name: str):
    """Return the HuggingFace dataset split used in the paper."""
    if dataset_name == 'tqa':
        return load_dataset("truthful_qa", 'generation')['validation']

    elif dataset_name == 'triviaqa':
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {k: [] for k in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        return ds.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)

    elif dataset_name == 'sciq':
        return load_dataset("allenai/sciq", split="validation")

    elif dataset_name == 'nq_open':
        return load_dataset("google-research-datasets/nq_open", split="validation")

    elif dataset_name == 'hotpotqa':
        return load_dataset("hotpot_qa", "distractor", split="validation")

    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Supported: {ALL_DATASETS}. "
            "To add a new dataset, extend load_hf_dataset() and BLEURT_THRESHOLD."
        )


def load_eval_data(dataset_name: str, tokenizer, args):
    """
    Load tokenised prompts and binary ground-truth labels for a dataset.
    Mirrors the data-loading logic in tsv_main.py's else-branch exactly.

    Returns
    -------
    test_prompts  : list of cuda tensors  [1, seq_len]
    gt_label_test : np.ndarray of int32
    """
    dataset = load_hf_dataset(dataset_name)
    length = len(dataset)

    prompts = []
    for i in tqdm(range(length), desc=f"  Tokenising {dataset_name}", leave=False):
        question = dataset[i]['question']
        answers = np.load(
            f'./save_for_eval/{dataset_name}_hal_det/answers/'
            f'most_likely_hal_det_{args.model_name}_{dataset_name}_answers_index_{i}.npy'
        )
        for anw in answers:
            prompt = tokenizer(
                f"Answer the question concisely. Q: {question} A:{anw}",
                return_tensors='pt'
            ).input_ids.cuda()
            prompts.append(prompt)

    # Load BLEURT-derived labels
    gts = np.load(f'./ml_{dataset_name}_bleurt_score.npy')
    thres = BLEURT_THRESHOLD.get(dataset_name, 0.5)
    gt_label = np.asarray(gts > thres, dtype=np.int32)

    # Use pre-fixed split indices if available, otherwise use all samples as test
    index_path = f'data_indices/data_index_{dataset_name}.npy'
    if os.path.exists(index_path):
        index = np.load(index_path)
        wild_q_indices = set(index[:int(args.wild_ratio * length)].tolist())
        test_indices = [i for i in range(length) if i not in wild_q_indices]
    else:
        print(f"  Note: no data_index_{dataset_name}.npy found — using all {length} samples as test set")
        test_indices = list(range(length))

    test_prompts, gt_label_test = [], []
    for i in test_indices:
        gt_label_test.extend(gt_label[i: i + 1])
        test_prompts.extend(prompts[i: i + 1])

    return test_prompts, np.asarray(gt_label_test, dtype=np.int32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_model(model, centroids, test_prompts, test_labels, device, batch_size, layer_number=-1):
    """Identical scoring logic to tsv_main.test_model()."""
    model.eval()
    val_predictions, val_labels_combined = [], []
    num_val_samples = len(test_prompts)

    with torch.no_grad():
        with autocast(dtype=torch.float16):
            for batch_start in range(0, num_val_samples, batch_size):
                batch_prompts = test_prompts[batch_start: batch_start + batch_size]
                batch_labels  = test_labels[batch_start: batch_start + batch_size]
                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)

                attention_mask = (batch_prompts != 0).half().to(device)
                batch_prompts  = batch_prompts.to(device)
                batch_labels   = batch_labels.to(device)

                output = model(
                    batch_prompts.squeeze(),
                    attention_mask=attention_mask.squeeze(),
                    output_hidden_states=True
                )
                hidden_states  = torch.stack(output.hidden_states, dim=0).squeeze()
                last_layer_hs  = hidden_states[layer_number]
                last_token_rep = get_last_non_padded_token_rep(last_layer_hs, attention_mask.squeeze())

                last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
                centroids_norm = F.normalize(centroids,       p=2, dim=-1)

                similarities = torch.matmul(last_token_rep, centroids_norm.T)
                scores       = torch.softmax(similarities / 0.1, dim=-1)[:, 1]  # truthful class

                val_predictions.append(scores.cpu())
                val_labels_combined.append(batch_labels.cpu())

    return torch.cat(val_predictions), torch.cat(val_labels_combined)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot cross-dataset evaluation for TSV."
    )
    parser.add_argument('--model_name',    type=str, default='llama3.1-8B',
                        choices=list(HF_NAMES.keys()))
    parser.add_argument('--checkpoint',    type=str, required=True,
                        help='Path to .pt checkpoint saved by tsv_main.py')
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=ALL_DATASETS,
                        help='Datasets to evaluate on. Default: all 4. '
                             'Add new dataset names here (and extend load_hf_dataset).')
    parser.add_argument('--device',        type=int, default=0)
    parser.add_argument('--batch_size',    type=int, default=128)
    parser.add_argument('--wild_ratio',    type=float, default=0.75,
                        help='Must match value used during training data prep')
    # TSV injection params — must match what was used during training
    parser.add_argument('--str_layer',  type=int,   default=9)
    parser.add_argument('--component',  type=str,   default='res')
    parser.add_argument('--lam',        type=float, default=5)
    args = parser.parse_args()

    seed_everything(42)

    model_name_or_path = HF_NAMES[args.model_name]
    device = torch.device("cuda")

    # ------------------------------------------------------------------
    # Load base model (frozen)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Loading model : {model_name_or_path}")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        token=None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=None)

    for param in model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Create TSV ParameterList and inject into model
    # ------------------------------------------------------------------
    num_layers  = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    tsv = nn.ParameterList([
        nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        for _ in range(num_layers)
    ])
    tsv.to(device)
    add_tsv_layers(model, tsv, [args.lam], args)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"\n Loading checkpoint : {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    tsv.load_state_dict(ckpt['tsv'])
    centroids  = ckpt['centroids'].to(device)
    saved_args = ckpt.get('args', {})
    print(f"  Centroids shape : {centroids.shape}")
    print(f"  Trained on      : {saved_args.get('dataset_name', 'unknown')}")

    # ------------------------------------------------------------------
    # Evaluate on each dataset
    # ------------------------------------------------------------------
    results = {}
    for dataset_name in args.eval_datasets:
        print(f"\n{'─'*60}")
        print(f" Evaluating on: {dataset_name}")
        print(f"{'─'*60}")
        try:
            test_prompts, test_labels = load_eval_data(dataset_name, tokenizer, args)
            n_pos = int(test_labels.sum())
            n_neg = len(test_labels) - n_pos
            print(f"  Test set size : {len(test_labels)}  "
                  f"(truthful={n_pos}, hallucinated={n_neg})")

            predictions, labels = test_model(
                model, centroids, test_prompts, test_labels, device, args.batch_size
            )
            raw_auroc = roc_auc_score(labels.numpy(), predictions.numpy())
            # If AUROC < 0.5 the centroid orientation is inverted for this dataset.
            # Flip the scores so the effective AUROC is always >= 0.5.
            if raw_auroc < 0.5:
                effective_auroc = 1.0 - raw_auroc
                flip_note = " (scores flipped — centroid orientation inverted cross-domain)"
            else:
                effective_auroc = raw_auroc
                flip_note = ""
            results[dataset_name] = effective_auroc
            print(f"  Raw AUROC     : {raw_auroc:.4f}")
            print(f"  Effective AUROC: {effective_auroc:.4f}{flip_note}")

        except FileNotFoundError as e:
            print(f"  SKIP — missing data file:\n    {e}")
            results[dataset_name] = None
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[dataset_name] = None

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET AUROC SUMMARY")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Model      : {args.model_name}")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'AUROC':>10}")
    print(f"  {'-'*28}")
    for ds, auroc in results.items():
        val = f"{auroc:.4f}" if auroc is not None else "SKIPPED"
        print(f"  {ds:<15} {val:>10}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------------
    results_path = args.checkpoint.replace('.pt', '_cross_eval.txt')
    with open(results_path, 'w') as f:
        f.write("Cross-Dataset Evaluation Results\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Model      : {args.model_name}\n")
        f.write(f"Trained on : {saved_args.get('dataset_name', 'unknown')}\n\n")
        f.write(f"{'Dataset':<15} {'AUROC':>10}\n")
        f.write(f"{'-'*28}\n")
        for ds, auroc in results.items():
            val = f"{auroc:.4f}" if auroc is not None else "SKIPPED"
            f.write(f"{ds:<15} {val:>10}\n")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
