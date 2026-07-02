#!/usr/bin/env bash
# Q1e: original EOGS with extra seeds {2024,3407} on all 7 AOIs (via patched EOGS_SEED env var).
# Purpose: show that the seed-variance finding is not specific to EOGS++.
set -uo pipefail

BASE=/root/autodl-tmp/eogs_v1
LOGDIR=$BASE/m2a_logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/q1e_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$LOG" "$LOGDIR/q1e_latest.log"
exec > >(tee -a "$LOG") 2>&1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eogsv1
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

data=$BASE/data
cd $BASE/src/gaussiansplatting
NUMIT=5000

echo "===== Q1E EOGS-V1 EXTRA SEEDS START $(date '+%F %T') ====="
for seed in 2024 3407; do
  export EOGS_SEED=$seed
  for scene in JAX_260 JAX_214 JAX_004 JAX_068 IARPA_001 IARPA_002 IARPA_003; do
    expname=m2a_eogsv1_${scene}_s${seed}
    outdir=$BASE/output/$expname
    echo "===== START $expname $(date '+%F %T') ====="
    if [[ -f "$outdir/${scene}_rdsm.tif" ]]; then
      echo "EXISTS, skip: $outdir"; continue
    fi
    if [[ ! -d "$outdir/test_opNone/ours_${NUMIT}/dsm" ]]; then
      python train.py -s ${data}/affine_models/${scene} --images ${data}/images/${scene} \
        --eval -m ${outdir} --sh_degree 0 --iterations ${NUMIT}
      python render.py -m ${outdir}
    fi
    dsm_name=$(ls ${outdir}/test_opNone/ours_${NUMIT}/dsm/ | sort -V | tail -n 1)
    python $BASE/scripts/eval/eval_dsm.py \
      --pred-dsm-path ${outdir}/test_opNone/ours_${NUMIT}/dsm/${dsm_name} \
      --gt-dir ${data}/truth/${scene} --out-dir ${outdir}/ --aoi-id ${scene}
    echo "===== DONE $expname $(date '+%F %T') ====="
  done
done
unset EOGS_SEED
echo "===== Q1E ALL DONE $(date '+%F %T') ====="
