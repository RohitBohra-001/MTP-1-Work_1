#!/usr/bin/env bash
# train_save.sh — Train TSV on TruthfulQA (tqa) and save checkpoint.
# Must be run AFTER gen_all.sh and gt_all.sh have completed.

set -eo pipefail

echo "============================================================"
echo " Training TSV on TQA — saving checkpoint to:"
echo "   ./checkpoints/tsv_tqa.pt"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python tsv_main.py \
    --model_name llama3.1-8B \
    --dataset_name tqa \
    --most_likely 1 \
    --save_checkpoint ./checkpoints/tsv_tqa.pt \
    2>&1 | tee train_tqa.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Training failed. Check train_tqa.log"
    exit 1
fi

echo ""
echo "Training complete. Checkpoint saved to ./checkpoints/tsv_tqa.pt"
echo "See train_tqa.log for epoch-by-epoch AUROC."
