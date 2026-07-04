#!/usr/bin/env bash
# One-factor TSDF arm: extract TSDF-fused DSMs from the EXISTING EOGS++ baseline
# checkpoints (no retraining), 7 scenes x 6 seeds. ~12 s per run.
set -uo pipefail
cd /root/autodl-tmp/eogs/EOGS2
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eogsplus
export PROJECT_ROOT=$PWD
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=0

for scene in JAX_260 JAX_214 JAX_004 JAX_068 IARPA_001 IARPA_002 IARPA_003; do
  sh=${scene//_/}
  for seed in 1337 2024 3407 5150 6001 7777; do
    if [[ $seed == 1337 ]]; then
      exp=phase1_eogsplus_rawrpc_${sh}_pan_3PAN_5000
    else
      exp=q1_baseline_s${seed}_${sh}_pan_3PAN_5000
    fi
    out=output/$exp/test_opNone/ours_5000/tsdf/${scene}_rdsm.tif
    if [[ -f $out ]]; then echo "EXISTS $exp"; continue; fi
    if python src/gaussiansplatting/tsdf.py experiments=eogsplus.yaml mode=3PAN \
        dataset=pan rpc_type=rpc_raw scene=$scene expname=$exp numiterations=5000 \
        run_tsdf=True ++seed=$seed >/dev/null 2>&1 && [[ -f $out ]]; then
      echo "OK $exp"
    else
      echo "FAIL $exp"
    fi
  done
done
echo "TSDF ALL DONE"
