#!/usr/bin/env bash
# cross_eval.sh — Zero-shot cross-dataset evaluation of a saved TSV checkpoint.
# Evaluates the TQA-trained checkpoint on all 4 datasets.
# Must be run AFTER train_save.sh has completed.

set -e

echo "============================================================"
echo " Cross-Dataset Evaluation"
echo " Checkpoint: ./checkpoints/tsv_tqa.pt"
echo " Datasets  : tqa triviaqa sciq nq_open"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python cross_eval.py \
    --model_name llama3.1-8B \
    --checkpoint ./checkpoints/tsv_tqa.pt \
    --eval_datasets tqa triviaqa sciq nq_open hotpotqa \
    2>&1 | tee cross_eval.log

echo ""
echo "Evaluation complete. See cross_eval.log for AUROC table."
echo "Results also saved to: ./checkpoints/tsv_tqa_cross_eval.txt"
