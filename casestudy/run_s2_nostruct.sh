#!/usr/bin/env bash
# S2: structure-protection ablation. phase1b-default WITHOUT structure protection, all 7 AOIs, seed=1337.
# baseline(=phase1_eogsplus_rawrpc_*) and ours(=phase3_phase1b_default_*/phase1b_structure_protected_JAX260) already exist.
set -Eeuo pipefail

ROOT=/root/autodl-tmp/eogs
REPO=$ROOT/EOGS2
OUTROOT=$ROOT/s2_ablation
LOGDIR=$OUTROOT/logs
mkdir -p "$LOGDIR"

LOG=$LOGDIR/s2_nostruct_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$LOG" "$LOGDIR/s2_latest.log"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eogsplus

unset EOGS_DEM_LOSS_PATH EOGS_DEM_LOSS_WEIGHT EOGS_DEM_LOSS_UNIFORM_WEIGHT EOGS_DEM_LOSS_EVIDENCE_WEIGHT
unset EOGS_DEM_LOSS_WDEM_PATH EOGS_DEM_LOSS_WWEAK_PATH

export PROJECT_ROOT="$REPO"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=0

# phase1b-default evidence weighting, structure protection OFF
export EOGS_VIEW_EVIDENCE_WEIGHTING=1
export EOGS_VIEW_EVIDENCE_PAN_ONLY=1
export EOGS_VIEW_EVIDENCE_FLOOR=0.60
export EOGS_VIEW_EVIDENCE_TEXTURE_Q=0.70
export EOGS_VIEW_EVIDENCE_BRIGHT_Q=0.15
export EOGS_VIEW_EVIDENCE_TEXTURE_WEIGHT=0.45
export EOGS_VIEW_EVIDENCE_BRIGHT_WEIGHT=0.25
export EOGS_VIEW_EVIDENCE_RESIDUAL_WEIGHT=0.30
export EOGS_VIEW_EVIDENCE_RESIDUAL_START=1500
export EOGS_VIEW_EVIDENCE_NORMALIZE=1
export EOGS_VIEW_EVIDENCE_BLEND=1.0
export EOGS_VIEW_STRUCTURE_PROTECT=0
export EOGS_VIEW_EVIDENCE_SAVE_EVERY=1000

SCENES=(JAX_260 JAX_214 JAX_004 JAX_068 IARPA_001 IARPA_002 IARPA_003)

echo "===== S2 NO-STRUCTURE ABLATION START $(date '+%F %T') ====="
for SCENE in "${SCENES[@]}"; do
  SHORT=${SCENE//_/}
  EXP=s2_nostruct_${SHORT}_pan_3PAN_5000
  WEIGHTDIR=$OUTROOT/weights/nostruct/$SCENE
  mkdir -p "$WEIGHTDIR"
  export EOGS_VIEW_EVIDENCE_SAVE_DIR="$WEIGHTDIR"
  OUT="$REPO/output/$EXP/test_opNone/ours_5000/rdsm/${SCENE}_rdsm.tif"
  echo "===== START $EXP $(date '+%F %T') ====="
  if [[ -f "$OUT" ]]; then
    echo "EXISTS, skip: $OUT"
  else
    python src/gaussiansplatting/full_eval_pan.py \
      experiments=eogsplus.yaml mode=3PAN dataset=pan rpc_type=rpc_raw \
      scene="$SCENE" expname="$EXP" numiterations=5000 run_tsdf=False
  fi
  echo "===== DONE $EXP $(date '+%F %T') ====="
done
echo "===== S2 NO-STRUCTURE ABLATION ALL DONE $(date '+%F %T') ====="
