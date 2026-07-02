#!/usr/bin/env bash
# M2a: original EOGS (mezzelfo) on all 7 AOIs, default config, seed as shipped.
# Runs train + render + official eval_dsm (produces registered DSM for unified eval).
set -uo pipefail

BASE=/root/autodl-tmp/eogs_v1
LOGDIR=$BASE/m2a_logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/m2a_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$LOG" "$LOGDIR/m2a_latest.log"
exec > >(tee -a "$LOG") 2>&1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eogsv1

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

data=$BASE/data
cd $BASE/src/gaussiansplatting

NUMIT=5000
echo "===== M2A EOGS-V1 START $(date '+%F %T') ====="
for scene in JAX_260 JAX_214 JAX_004 JAX_068 IARPA_001 IARPA_002 IARPA_003; do
  expname=m2a_eogsv1_${scene}
  outdir=$BASE/output/$expname
  echo "===== START $expname $(date '+%F %T') ====="
  if [[ -d "$outdir/test_opNone/ours_${NUMIT}/dsm" ]]; then
    echo "EXISTS, skip training: $outdir"
  else
    python train.py \
      -s ${data}/affine_models/${scene} \
      --images ${data}/images/${scene} \
      --eval \
      -m ${outdir} \
      --sh_degree 0 \
      --iterations ${NUMIT}
    python render.py -m ${outdir}
  fi
  dsm_name=$(ls ${outdir}/test_opNone/ours_${NUMIT}/dsm/ | sort -V | tail -n 1)
  echo "DSM: $dsm_name"
  python $BASE/scripts/eval/eval_dsm.py \
    --pred-dsm-path ${outdir}/test_opNone/ours_${NUMIT}/dsm/${dsm_name} \
    --gt-dir ${data}/truth/${scene} \
    --out-dir ${outdir}/ \
    --aoi-id ${scene}
  echo "===== DONE $expname $(date '+%F %T') ====="
done
echo "===== M2A EOGS-V1 ALL DONE $(date '+%F %T') ====="
