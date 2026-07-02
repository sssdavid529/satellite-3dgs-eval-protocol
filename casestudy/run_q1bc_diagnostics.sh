#!/usr/bin/env bash
# Q1b: extra seeds {5150,6001,7777} for JAX_260/JAX_214, baseline & ours (5000 it) -> 12 runs
# Q1c: convergence diagnostic, 15000 it, JAX_260, baseline & ours, seeds {1337,2024} -> 4 runs
set -Eeuo pipefail

ROOT=/root/autodl-tmp/eogs
REPO=$ROOT/EOGS2
OUTROOT=$ROOT/q1_seeds
LOGDIR=$OUTROOT/logs
mkdir -p "$LOGDIR"

LOG=$LOGDIR/q1bc_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$LOG" "$LOGDIR/q1bc_latest.log"
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

clear_evidence() {
  unset EOGS_VIEW_EVIDENCE_WEIGHTING EOGS_VIEW_EVIDENCE_PAN_ONLY EOGS_VIEW_EVIDENCE_FLOOR \
        EOGS_VIEW_EVIDENCE_TEXTURE_Q EOGS_VIEW_EVIDENCE_BRIGHT_Q EOGS_VIEW_EVIDENCE_TEXTURE_WEIGHT \
        EOGS_VIEW_EVIDENCE_BRIGHT_WEIGHT EOGS_VIEW_EVIDENCE_RESIDUAL_WEIGHT EOGS_VIEW_EVIDENCE_RESIDUAL_START \
        EOGS_VIEW_EVIDENCE_NORMALIZE EOGS_VIEW_EVIDENCE_BLEND \
        EOGS_VIEW_STRUCTURE_PROTECT EOGS_VIEW_STRUCTURE_EDGE_Q EOGS_VIEW_STRUCTURE_LOW EOGS_VIEW_STRUCTURE_HIGH \
        EOGS_VIEW_STRUCTURE_POWER EOGS_VIEW_STRUCTURE_KERNEL EOGS_VIEW_EVIDENCE_SAVE_EVERY EOGS_VIEW_EVIDENCE_SAVE_DIR || true
}

set_phase1b_defaults() {
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
  export EOGS_VIEW_STRUCTURE_PROTECT=1
  export EOGS_VIEW_STRUCTURE_EDGE_Q=0.78
  export EOGS_VIEW_STRUCTURE_LOW=0.20
  export EOGS_VIEW_STRUCTURE_HIGH=0.58
  export EOGS_VIEW_STRUCTURE_POWER=1.20
  export EOGS_VIEW_STRUCTURE_KERNEL=5
  export EOGS_VIEW_EVIDENCE_SAVE_EVERY=100000
}

run_one() {
  local cfg="$1" scene="$2" seed="$3" iters="$4"
  local short=${scene//_/}
  local exp=q1_${cfg}_s${seed}_${short}_pan_3PAN_${iters}
  local out="$REPO/output/$exp/test_opNone/ours_${iters}/rdsm/${scene}_rdsm.tif"
  echo "===== START $exp $(date '+%F %T') ====="
  if [[ -f "$out" ]]; then
    echo "EXISTS, skip: $out"
  else
    python src/gaussiansplatting/full_eval_pan.py \
      experiments=eogsplus.yaml mode=3PAN dataset=pan rpc_type=rpc_raw \
      scene="$scene" expname="$exp" numiterations="$iters" run_tsdf=False ++seed="$seed"
  fi
  echo "===== DONE $exp $(date '+%F %T') ====="
}

echo "===== Q1B EXTRA SEEDS START $(date '+%F %T') ====="
for seed in 5150 6001 7777; do
  for scene in JAX_260 JAX_214; do
    clear_evidence
    run_one baseline "$scene" "$seed" 5000
    set_phase1b_defaults
    run_one ours "$scene" "$seed" 5000
  done
done

echo "===== Q1C CONVERGENCE 15K START $(date '+%F %T') ====="
for seed in 1337 2024; do
  clear_evidence
  run_one baseline JAX_260 "$seed" 15000
  set_phase1b_defaults
  run_one ours JAX_260 "$seed" 15000
done
echo "===== Q1BC ALL DONE $(date '+%F %T') ====="
