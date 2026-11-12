#!/usr/bin/env bash
# gen_all.sh — Generate LLM responses for all 4 datasets (sequentially, GPU 0)
# Each dataset runs to completion before the next starts.
# Runtime: ~1-3 hours per dataset depending on dataset size.

set -eo pipefail

DATASETS=("tqa" "triviaqa" "sciq" "nq_open" "hotpotqa")

for DS in "${DATASETS[@]}"; do
    echo "============================================================"
    echo " Generating responses for: ${DS}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=0 python tsv_main.py \
        --model_name llama3.1-8B \
        --dataset_name "${DS}" \
        --most_likely 1 \
        --num_gene 1 \
        --gene 1 \
        2>&1 | tee "gen_${DS}.log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Generation failed for ${DS}. Check gen_${DS}.log"
        exit 1
    fi
    echo " Done: ${DS}"
done

echo ""
echo "All generation jobs complete."
