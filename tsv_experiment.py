import os
os.environ["PYTORCH_NO_CUDA_NVML"] = "1"
import sys
import json
import csv
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_utils import (
    get_last_non_padded_token_rep,
    compute_ot_loss_cos,
    compute_entropy,
    update_centroids_ema,
    update_centroids_ema_hard,
    collate_fn
)
from llm_layers import add_tsv_layers
from sinkhorn_knopp import SinkhornKnopp_imb


HF_NAMES = {
    'llama3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'qwen2.5-7B': 'Qwen/Qwen2.5-7B'
}


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_set(model, centroids, prompts, labels, device, batch_size, layer_number=-1):
    """Evaluates representations against centroids and computes both raw and effective AUROC."""
    model.eval()
    val_predictions = []
    val_labels_combined = []
    num_samples = len(prompts)

    with torch.no_grad():
        with autocast(dtype=torch.float16):
            for batch_start in range(0, num_samples, batch_size):
                batch_prompts = prompts[batch_start:batch_start + batch_size]
                batch_labels = labels[batch_start:batch_start + batch_size]
                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)

                attention_mask = (batch_prompts != 0).half().to(device)
                batch_prompts = batch_prompts.to(device)
                batch_labels = batch_labels.to(device)

                output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                last_layer_hidden_state = output.hidden_states[layer_number]
                last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())

                last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
                norm_centroids = F.normalize(centroids, p=2, dim=-1)

                similarities = torch.matmul(last_token_rep, norm_centroids.T)
                similarity_scores = torch.softmax(similarities / 0.1, dim=-1)
                similarity_scores = similarity_scores[:, 1]  # Score for truthful class

                val_predictions.append(similarity_scores.cpu())
                val_labels_combined.append(batch_labels.cpu())

    val_predictions = torch.cat(val_predictions).numpy()
    val_labels_combined = torch.cat(val_labels_combined).numpy()

    # If only one class present in split, handle gracefully
    if len(np.unique(val_labels_combined)) < 2:
        return 0.5, 0.5

    raw_auroc = roc_auc_score(val_labels_combined, val_predictions)
    effective_auroc = max(raw_auroc, 1.0 - raw_auroc)
    return float(raw_auroc), float(effective_auroc)


def get_pseudo_labeled_data(model, prompts, labels, batch_size, centroids, sinkhorn, num_selected_data, cls_dist, args, method='sinkhorn'):
    all_embeddings = []
    num_samples = len(prompts)

    with torch.no_grad():
        with autocast(dtype=torch.float16):
            for batch_start in tqdm(range(0, num_samples, batch_size), desc="Pseudo-labeling", leave=False):
                batch_prompts = prompts[batch_start: batch_start + batch_size]
                batch_labels = labels[batch_start: batch_start + batch_size]
                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)
                attention_mask = (batch_prompts != 0).half().to(batch_prompts.device)
                batch_prompts = batch_prompts.cuda()
                batch_labels = batch_labels.cuda()
                attention_mask = attention_mask.to(batch_prompts.device)

                output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                last_layer_hidden_state = output.hidden_states[-1]

                last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())
                all_embeddings.append(last_token_rep.detach())

            all_embeddings = F.normalize(torch.concat(all_embeddings), p=2, dim=-1)

            if method == 'sinkhorn':
                pseudo_label = sinkhorn(all_embeddings, centroids)
            elif method == 'simple':
                # Missing baseline: standard cosine softmax pseudo-labeling without Optimal Transport
                similarities = torch.matmul(all_embeddings, centroids.T) / args.cos_temp
                pseudo_label = F.softmax(similarities, dim=-1)
            else:
                raise ValueError(f"Unknown pseudo_label_method: {method}")

            selected_indices = compute_entropy(all_embeddings, centroids, pseudo_label, num_selected_data, cls_dist, args)
            selected_labels_soft = pseudo_label[selected_indices].detach()

    del all_embeddings
    torch.cuda.empty_cache()

    return selected_indices, selected_labels_soft


def train_and_evaluate(model, optimizer, device, split_prompts, split_labels, args):
    layer_number = -1
    scaler = GradScaler()

    test_prompts, val_prompts, train_prompts, exemplar_prompts = split_prompts
    test_labels, val_labels, train_labels, exemplar_labels = split_labels

    batch_size = args.batch_size
    num_exemplars = len(exemplar_prompts)

    # Sinkhorn setup
    args.num_iters_sk = 3
    args.epsilon_sk = 0.05
    ex_hallu = (num_exemplars - exemplar_labels[:num_exemplars].sum()) / num_exemplars
    ex_true = (exemplar_labels[:num_exemplars].sum()) / num_exemplars
    cls_dist = torch.tensor([ex_hallu, ex_true]).float().cuda().view(-1, 1)
    sinkhorn = SinkhornKnopp_imb(args, cls_dist)

    # Random Centroid initialization
    centroids = torch.randn((2, model.config.hidden_size)).half().cuda()
    centroids = F.normalize(centroids, p=2, dim=1)

    exemplar_prompts_raw = list(exemplar_prompts)
    exemplar_labels_raw = np.array(exemplar_labels)
    exemplar_prompts_collated, exemplar_labels_collated = collate_fn(exemplar_prompts_raw, exemplar_labels_raw)

    # Tracking states across entire training
    best_val_effective_auroc = -1.0
    best_val_epoch = -1
    best_val_tsv_state = None
    best_val_centroids = None
    best_val_test_effective_auroc = -1.0

    best_test_snooped_auroc = -1.0
    best_test_snooped_epoch = -1

    # Tracking Phase 1 states explicitly under all 3 selection rules
    p1_best_val_effective_auroc = -1.0
    p1_best_val_epoch = -1
    p1_best_val_tsv_state = None
    p1_best_val_centroids = None
    p1_best_val_test_effective_auroc = -1.0

    p1_best_test_snooped_auroc = -1.0
    p1_best_test_snooped_epoch = -1

    history = []

    # =========================================================================
    # PHASE 1: Exemplar Training
    # =========================================================================
    print(f"\n[Phase 1] Training on {num_exemplars} exemplars for {args.init_num_epochs} epochs...")
    for epoch in range(args.init_num_epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for batch_start in range(0, num_exemplars, batch_size):
            b_prompts = exemplar_prompts_collated[batch_start: batch_start + batch_size]
            b_labels = exemplar_labels_collated[batch_start: batch_start + batch_size]

            attention_mask = (b_prompts != 0).half().to(device)
            b_prompts = b_prompts.to(device)
            b_labels = b_labels.to(device)

            with autocast(dtype=torch.float16):
                output = model(b_prompts.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                last_layer_hidden_state = output.hidden_states[layer_number]
                last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())

                b_labels_oh = F.one_hot(b_labels, num_classes=2)
                loss, _ = compute_ot_loss_cos(last_token_rep, centroids, b_labels_oh, batch_size, args)

                total += b_labels.size(0)
                with torch.no_grad():
                    centroids = update_centroids_ema_hard(centroids, last_token_rep, b_labels_oh, args)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item() * b_labels.size(0)

        epoch_loss = running_loss / max(total, 1)

        # Evaluate on Val (and conditionally on Test if not in fast_eval mode)
        val_raw, val_eff = evaluate_set(model, centroids, val_prompts, val_labels, device, batch_size, layer_number)
        if not args.fast_eval:
            test_raw, test_eff = evaluate_set(model, centroids, test_prompts, test_labels, device, batch_size, layer_number)
        else:
            test_raw, test_eff = 0.0, 0.0

        if val_eff > best_val_effective_auroc:
            best_val_effective_auroc = val_eff
            best_val_epoch = epoch + 1
            best_val_tsv_state = {k: v.clone() for k, v in model.named_parameters() if 'tsv' in k.lower()}
            best_val_centroids = centroids.clone()
            best_val_test_effective_auroc = test_eff

        if test_eff > best_test_snooped_auroc:
            best_test_snooped_auroc = test_eff
            best_test_snooped_epoch = epoch + 1

        # Phase 1 specific tracking
        if val_eff > p1_best_val_effective_auroc:
            p1_best_val_effective_auroc = val_eff
            p1_best_val_epoch = epoch + 1
            p1_best_val_tsv_state = {k: v.clone() for k, v in model.named_parameters() if 'tsv' in k.lower()}
            p1_best_val_centroids = centroids.clone()
            p1_best_val_test_effective_auroc = test_eff

        if test_eff > p1_best_test_snooped_auroc:
            p1_best_test_snooped_auroc = test_eff
            p1_best_test_snooped_epoch = epoch + 1

        history.append({
            'epoch': epoch + 1,
            'phase': 1,
            'loss': float(epoch_loss),
            'val_auroc': float(val_eff),
            'test_auroc': float(test_eff)
        })

    # Phase 1 completion tracking (CRITICAL: NEVER overwrite live model parameters mid-training)
    p1_last_tsv_state = {k: v.clone() for k, v in model.named_parameters() if 'tsv' in k.lower()}
    p1_last_centroids = centroids.clone()
    phase1_last_val_eff = history[-1]['val_auroc']
    if not args.fast_eval:
        phase1_last_test_eff = history[-1]['test_auroc']
    else:
        phase1_last_test_eff = 0.0

    print(f"[Phase 1 Finished] Last-Epoch Val: {phase1_last_val_eff:.4f} | Best Val Epoch: {p1_best_val_epoch} (Val: {p1_best_val_effective_auroc:.4f})")
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: Pseudo-Labeling & Augmented Retraining
    # =========================================================================
    if args.aug_num_epochs > 0:
        print(f"\n[Phase 2] Pseudo-labeling (Method: {args.pseudo_label_method}) & retraining for {args.aug_num_epochs} epochs...")
        with torch.no_grad():
            selected_indices, selected_labels_soft = get_pseudo_labeled_data(
                model, train_prompts, train_labels, batch_size, centroids, sinkhorn,
                args.num_selected_data, cls_dist, args, method=args.pseudo_label_method
            )

        selected_prompts = [train_prompts[i] for i in selected_indices]
        augmented_prompts = selected_prompts + exemplar_prompts_raw

        exemplar_label_tensor = torch.tensor(exemplar_labels_raw).cuda()
        exemplar_labels_oh = F.one_hot(exemplar_label_tensor.to(torch.int64), num_classes=2)
        augmented_labels = torch.concat((selected_labels_soft, exemplar_labels_oh.clone().cuda()))

        num_aug_samples = len(augmented_prompts)

        with autocast(dtype=torch.float16):
            for epoch in range(args.aug_num_epochs):
                model.train()
                running_loss = 0.0
                total = 0

                for batch_start in range(0, num_aug_samples, batch_size):
                    batch_p = augmented_prompts[batch_start: batch_start + batch_size]
                    batch_l = augmented_labels[batch_start: batch_start + batch_size]
                    batch_p, batch_l = collate_fn(batch_p, batch_l)

                    attention_mask = (batch_p != 0).half().to(device)
                    batch_p = batch_p.to(device)
                    batch_l = batch_l.to(device)

                    output = model(batch_p.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                    last_layer_hidden_state = output.hidden_states[layer_number]
                    last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())

                    loss, _ = compute_ot_loss_cos(last_token_rep, centroids, batch_l, batch_size, args)

                    total += batch_l.size(0)
                    with torch.no_grad():
                        centroids = update_centroids_ema(centroids, last_token_rep, batch_l.half(), args)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    running_loss += loss.item() * batch_l.size(0)

                epoch_loss = running_loss / max(total, 1)
                global_epoch = args.init_num_epochs + epoch + 1

                val_raw, val_eff = evaluate_set(model, centroids, val_prompts, val_labels, device, batch_size, layer_number)
                if not args.fast_eval:
                    test_raw, test_eff = evaluate_set(model, centroids, test_prompts, test_labels, device, batch_size, layer_number)
                else:
                    test_raw, test_eff = 0.0, 0.0

                if val_eff > best_val_effective_auroc:
                    best_val_effective_auroc = val_eff
                    best_val_epoch = global_epoch
                    best_val_tsv_state = {k: v.clone() for k, v in model.named_parameters() if 'tsv' in k.lower()}
                    best_val_centroids = centroids.clone()
                    best_val_test_effective_auroc = test_eff

                if test_eff > best_test_snooped_auroc:
                    best_test_snooped_auroc = test_eff
                    best_test_snooped_epoch = global_epoch

                history.append({
                    'epoch': global_epoch,
                    'phase': 2,
                    'loss': float(epoch_loss),
                    'val_auroc': float(val_eff),
                    'test_auroc': float(test_eff)
                })

    # Save last epoch state
    last_tsv_state = {k: v.clone() for k, v in model.named_parameters() if 'tsv' in k.lower()}
    last_centroids = centroids.clone()

    # If fast_eval, evaluate Phase 1 checkpoints now at the end (model training is 100% complete)
    if args.fast_eval:
        if p1_best_val_tsv_state is not None:
            for k, v in model.named_parameters():
                if 'tsv' in k.lower() and k in p1_best_val_tsv_state:
                    v.data.copy_(p1_best_val_tsv_state[k])
            _, p1_best_val_test_effective_auroc = evaluate_set(
                model, p1_best_val_centroids, test_prompts, test_labels, device, batch_size, layer_number
            )
        else:
            p1_best_val_test_effective_auroc = 0.0
        p1_best_test_snooped_auroc = p1_best_val_test_effective_auroc

        if p1_last_tsv_state is not None:
            for k, v in model.named_parameters():
                if 'tsv' in k.lower() and k in p1_last_tsv_state:
                    v.data.copy_(p1_last_tsv_state[k])
            _, phase1_last_test_eff = evaluate_set(
                model, p1_last_centroids, test_prompts, test_labels, device, batch_size, layer_number
            )
        else:
            phase1_last_test_eff = 0.0

    # Re-evaluate the best-validation model to ensure exact test performance
    if best_val_tsv_state is not None:
        for k, v in model.named_parameters():
            if 'tsv' in k.lower() and k in best_val_tsv_state:
                v.data.copy_(best_val_tsv_state[k])
        val_selected_test_raw, val_selected_test_eff = evaluate_set(
            model, best_val_centroids, test_prompts, test_labels, device, batch_size, layer_number
        )
    else:
        val_selected_test_eff = 0.0
        best_val_centroids = centroids

    if args.fast_eval:
        best_test_snooped_auroc = val_selected_test_eff
        best_test_snooped_epoch = best_val_epoch

    # Evaluate last-epoch test performance
    if last_tsv_state is not None:
        for k, v in model.named_parameters():
            if 'tsv' in k.lower() and k in last_tsv_state:
                v.data.copy_(last_tsv_state[k])
        last_test_raw, last_test_eff = evaluate_set(
            model, last_centroids, test_prompts, test_labels, device, batch_size, layer_number
        )
    else:
        last_test_eff = val_selected_test_eff

    summary = {
        # Full Pipeline (Phase 1+2) metrics
        'val_selected_test_auroc': float(val_selected_test_eff),
        'val_selected_epoch': int(best_val_epoch),
        'best_val_auroc': float(best_val_effective_auroc),
        'test_snooped_best_auroc': float(best_test_snooped_auroc),
        'test_snooped_best_epoch': int(best_test_snooped_epoch),
        'last_epoch_test_auroc': float(last_test_eff),

        # Phase 1 metrics under all 3 selection rules
        'phase1_val_selected_test_auroc': float(p1_best_val_test_effective_auroc),
        'phase1_snooped_test_auroc': float(p1_best_test_snooped_auroc),
        'phase1_last_epoch_test_auroc': float(phase1_last_test_eff),
        'phase1_test_auroc': float(phase1_last_test_eff),  # legacy compatibility
        'phase1_val_auroc': float(phase1_last_val_eff),

        # Fair Phase 2 Gains & Leakage
        'phase2_gain_honest': float(val_selected_test_eff - p1_best_val_test_effective_auroc),
        'phase2_gain_snooped': float(best_test_snooped_auroc - p1_best_test_snooped_auroc),
        'phase2_gain_val_selected': float(val_selected_test_eff - phase1_last_test_eff),
        'leakage_inflation': float(best_test_snooped_auroc - val_selected_test_eff),
    }

    return summary, best_val_centroids, best_val_tsv_state, history


def main():
    parser = argparse.ArgumentParser(description="TSV Rigorous Thesis Experiment Runner")
    parser.add_argument('--experiment_id', type=str, default='E2_seed_variance', help='Experiment ID for tracking')
    parser.add_argument('--model_name', type=str, default='llama3.1-8B')
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--str_layer', type=int, default=9)
    parser.add_argument('--component', type=str, default='res', choices=['res', 'mlp', 'attn'])
    parser.add_argument('--lam', type=float, default=5.0)
    parser.add_argument('--num_exemplars', type=int, default=32)
    parser.add_argument('--num_selected_data', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--init_num_epochs', type=int, default=20)
    parser.add_argument('--aug_num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--cos_temp', type=float, default=0.1)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--wild_ratio', type=float, default=0.75)
    parser.add_argument('--thres_gt', type=float, default=None)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--fast_eval', type=int, default=1, help='Fast evaluation: evaluate test set only 3 times at key checkpoints')
    parser.add_argument('--pseudo_label_method', type=str, default='sinkhorn', choices=['sinkhorn', 'simple'])
    parser.add_argument('--hf_token', type=str, default=os.environ.get('HF_TOKEN', ''), help='HuggingFace access token')
    parser.add_argument('--save_checkpoint', type=str, default=None)
    parser.add_argument('--results_jsonl', type=str, default='./results/all_experiments.jsonl')
    parser.add_argument('--results_csv', type=str, default='./results/summary_table.csv')
    args = parser.parse_args()

    # Enforce reproducibility
    seed_everything(args.seed)

    # Threshold defaults matching repo
    if args.thres_gt is None:
        args.thres_gt = 0.5 if args.dataset_name in ['tqa', 'triviaqa'] else 0.2

    print("=" * 70)
    print(f" [TSV EXPERIMENT] ID: {args.experiment_id} | Dataset: {args.dataset_name} | Seed: {args.seed}")
    print(f" Layer: {args.str_layer} | Lambda: {args.lam} | N: {args.num_exemplars} | K: {args.num_selected_data}")
    print(f" Method: {args.pseudo_label_method} | BLEURT Threshold: {args.thres_gt}")
    print("=" * 70)

    # Load dataset structure
    model_name_or_path = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=args.hf_token)

    if args.dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif args.dataset_name == 'sciq':
        dataset = load_dataset("allenai/sciq", split="validation")
    elif args.dataset_name == 'nq_open':
        dataset = load_dataset("google-research-datasets/nq_open", split="validation")
    elif args.dataset_name == 'hotpotqa':
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    else:
        raise ValueError(f"Invalid dataset {args.dataset_name}")

    # Build prompts from cached model answers
    length = len(dataset)
    prompts = []
    for i in range(length):
        question = dataset[i]['question']
        answers = np.load(f'./save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
        for anw in answers:
            prompt = tokenizer(
                f"Answer the question concisely. Q: {question} A:{anw}",
                return_tensors='pt'
            ).input_ids.cuda()
            prompts.append(prompt)

    # Load cached BLEURT scores & create binary ground-truth labels
    gts = np.load(f'./ml_{args.dataset_name}_bleurt_score.npy')
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)

    # Split creation with rigorous zero-leakage separation
    index_file = f'data_indices/data_index_{args.dataset_name}.npy'
    if os.path.exists(index_file):
        index = np.load(index_file)
    else:
        rng = np.random.RandomState(42)
        index = rng.permutation(length)

    wild_q_indices = index[:int(args.wild_ratio * length)]
    test_indices = [i for i in range(length) if i not in wild_q_indices]

    # Carve out 100 samples from the training pool as held-out validation
    val_size = min(args.val_size, len(wild_q_indices) // 4)
    val_indices = wild_q_indices[-val_size:]
    training_pool = wild_q_indices[:-val_size]

    # Exemplars: if 32 and default file exists, use it; otherwise sample deterministically from training_pool
    exemplar_file = f'data_indices/exemplar_idx_{args.dataset_name}.npy'
    if args.num_exemplars == 32 and os.path.exists(exemplar_file) and args.seed == 42:
        exemplar_indices = np.load(exemplar_file)
    else:
        rng_seed = np.random.RandomState(args.seed)
        exemplar_indices = rng_seed.choice(training_pool, size=args.num_exemplars, replace=False)

    exemplar_set = set(exemplar_indices)
    unlabeled_train_indices = [i for i in training_pool if i not in exemplar_set]

    # Assemble prompt and label splits
    test_prompts = [prompts[i] for i in test_indices]
    test_labels = gt_label[test_indices]

    val_prompts = [prompts[i] for i in val_indices]
    val_labels = gt_label[val_indices]

    train_prompts = [prompts[i] for i in unlabeled_train_indices]
    train_labels = gt_label[unlabeled_train_indices]

    exemplar_prompts = [prompts[i] for i in exemplar_indices]
    exemplar_labels = gt_label[exemplar_indices]

    print(f"Data Splits: Test={len(test_prompts)}, Val={len(val_prompts)}, Unlabeled Train={len(train_prompts)}, Exemplars={len(exemplar_prompts)}")

    # Load frozen model & insert TSV layer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        token=args.hf_token
    )
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    tsv = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size), requires_grad=True) for _ in range(num_layers)]).to(device)
    add_tsv_layers(model, tsv, [args.lam], args)
    optimizer = torch.optim.AdamW(list(tsv.parameters()), lr=args.lr)

    split_prompts = [test_prompts, val_prompts, train_prompts, exemplar_prompts]
    split_labels = [test_labels, val_labels, train_labels, exemplar_labels]

    summary, best_centroids, best_val_tsv_state, history = train_and_evaluate(model, optimizer, device, split_prompts, split_labels, args)

    print("\n" + "=" * 70)
    print(f" EXPERIMENT COMPLETE: {args.experiment_id}")
    print(f" Honest Val-Selected Test AUROC: {summary['val_selected_test_auroc']:.4f} (at Epoch {summary['val_selected_epoch']})")
    print(f" Snooped Best Test AUROC       : {summary['test_snooped_best_auroc']:.4f} (at Epoch {summary['test_snooped_best_epoch']})")
    print(f" Leakage Inflation Gap         : +{summary['leakage_inflation']:.4f}")
    print(f" Last-Epoch Test AUROC         : {summary['last_epoch_test_auroc']:.4f}")
    print(f" Phase 1 (Exemplars only)      : {summary['phase1_test_auroc']:.4f}")
    print(f" Phase 2 Real Gain             : {summary['phase2_gain_val_selected']:+.4f}")
    print("=" * 70 + "\n")

    # Record complete run metadata
    run_record = {
        'experiment_id': args.experiment_id,
        'dataset': args.dataset_name,
        'seed': args.seed,
        'str_layer': args.str_layer,
        'lam': args.lam,
        'num_exemplars': args.num_exemplars,
        'num_selected_data': args.num_selected_data,
        'pseudo_label_method': args.pseudo_label_method,
        'thres_gt': args.thres_gt,
        **summary
    }

    # Save curve history (loss, val auroc, test auroc per epoch)
    curves_dir = './results/curves'
    os.makedirs(curves_dir, exist_ok=True)
    curve_file = f"{curves_dir}/{args.experiment_id}_{args.dataset_name}_seed_{args.seed}_layer_{args.str_layer}.json"
    with open(curve_file, 'w') as f:
        json.dump(history, f, indent=2)

    # Save to JSONL
    os.makedirs(os.path.dirname(os.path.abspath(args.results_jsonl)), exist_ok=True)
    with open(args.results_jsonl, 'a') as f:
        f.write(json.dumps(run_record) + "\n")

    # Save / append to CSV summary table
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    csv_exists = os.path.exists(args.results_csv)
    fieldnames = list(run_record.keys())
    with open(args.results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(run_record)

    # Optional checkpoint save (strictly saves the val-selected checkpoint)
    if args.save_checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_checkpoint)), exist_ok=True)
        if best_val_tsv_state is not None:
            for k, v in model.named_parameters():
                if 'tsv' in k.lower() and k in best_val_tsv_state:
                    v.data.copy_(best_val_tsv_state[k])
        torch.save({
            'tsv': tsv.state_dict(),
            'centroids': best_centroids.cpu(),
            'summary': summary,
            'args': vars(args)
        }, args.save_checkpoint)
        print(f"Saved val-selected checkpoint to {args.save_checkpoint}")


if __name__ == '__main__':
    main()

