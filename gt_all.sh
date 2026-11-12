#!/usr/bin/env bash
# gt_all.sh — Compute BLEURT ground-truth scores for all 4 datasets (sequentially, GPU 0)
# Must be run AFTER gen_all.sh has completed for all datasets.

set -eo pipefail

DATASETS=("tqa" "triviaqa" "sciq" "nq_open" "hotpotqa")

for DS in "${DATASETS[@]}"; do
    echo "============================================================"
    echo " Computing BLEURT scores for: ${DS}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=0 python tsv_main.py \
        --model_name llama3.1-8B \
        --dataset_name "${DS}" \
        --most_likely 1 \
        --num_gene 1 \
        --generate_gt 1 \
        2>&1 | tee "gt_${DS}.log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: BLEURT scoring failed for ${DS}. Check gt_${DS}.log"
        exit 1
    fi
    echo " Done: ${DS}"
done

echo ""
echo "All BLEURT scoring jobs complete."
echo "Saved files: ml_tqa_bleurt_score.npy, ml_triviaqa_bleurt_score.npy,"
echo "             ml_sciq_bleurt_score.npy, ml_nq_open_bleurt_score.npy"
