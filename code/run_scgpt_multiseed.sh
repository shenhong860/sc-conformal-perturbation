#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cw1ot
export LD_LIBRARY_PATH=/opt/miniconda3/envs/cw1ot/lib
export PYTHONPATH=/mnt/d/guo/CW1OT/code/scgpt_shims
cd /mnt/d/guo/CW1OT/code
mkdir -p /mnt/d/guo/CW1OT/results/scgpt_norman
for s in 0 1 2 3 4; do
  echo "=== seed $s ==="
  python -u scgpt_finetune_norman.py $s
done
echo "=== aggregation ==="
python -u run_scgpt_multiseed_agg.py
