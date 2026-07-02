#!/usr/bin/env bash
# M2b: render DSMs from official Sat-NeRF checkpoints (4 JAX scenes x 2 variants),
# then register each with the EOGS-family eval_dsm.py (same registration as M2a).
# NeRF inference runs in satnerf310 (numpy<2, no gdal); registration runs in eogsv1.
# create_satnerf_dsm.py crashes at its final gdal-based MAE step - tolerated; DSM is written before that.
set -uo pipefail

SN=/root/autodl-tmp/satnerf
EV=$SN/ev2022
LOGDIR=$SN/m2b_logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/m2b_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$LOG" "$LOGDIR/m2b_latest.log"
exec > >(tee -a "$LOG") 2>&1

source /root/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=0

echo "===== M2B SATNERF CKPT START $(date '+%F %T') ====="
for scene in JAX_260 JAX_214 JAX_004 JAX_068; do
  for variant in "Sat-NeRF+DS" "Sat-NeRF"; do
    tag="${scene}_${variant//+/_}"
    ckptdir="$EV/pretrained_unzip/$scene/$variant"
    if [[ ! -d "$ckptdir" ]]; then
      echo "NO CKPT $ckptdir, skip"; continue
    fi
    # detect epoch number from checkpoint filename (epoch=N.ckpt -> N+1)
    ckfile=$(ls "$ckptdir"/epoch=*.ckpt 2>/dev/null | head -1)
    epn=$(basename "$ckfile" | sed -E 's/epoch=([0-9]+).*/\1/')
    ep=$((epn + 1))
    outdir="$SN/m2b_out/$tag"
    dsm_exists=$(find "$outdir" -name '*_dsm_epoch*.tif' 2>/dev/null | head -1)
    echo "===== START $tag (epoch $ep) $(date '+%F %T') ====="
    if [[ -z "$dsm_exists" ]]; then
      conda activate satnerf310
      timeout 900 python $SN/create_satnerf_dsm.py "$variant" \
        "$EV/pretrained_unzip/$scene" "$outdir" "$ep" \
        "$EV/pretrained_unzip/$scene" \
        "$EV/dataset_unzip/root_dir/crops_rpcs_ba_v2/$scene" \
        "$EV/dataset_unzip/DFC2019/Track3-RGB-crops/$scene" \
        "$EV/dataset_unzip/DFC2019/Track3-Truth" || echo "(inference exited nonzero - checking DSM file)"
      conda deactivate
    else
      echo "DSM exists, skip inference"
    fi
    dsm=$(find "$outdir" -name '*_dsm_epoch*.tif' 2>/dev/null | head -1)
    if [[ -z "$dsm" ]]; then
      echo "DSM_MISSING for $tag"; continue
    fi
    echo "DSM: $dsm"
    regdir="$SN/m2b_out/registered/$tag"
    if [[ ! -f "$regdir/${scene}_rdsm.tif" ]]; then
      mkdir -p "$regdir"
      conda activate eogsv1
      python /root/autodl-tmp/eogs_v1/scripts/eval/eval_dsm.py \
        --pred-dsm-path "$dsm" \
        --gt-dir /root/autodl-tmp/eogs_v1/data/truth/$scene \
        --out-dir "$regdir" \
        --aoi-id "$scene" || echo "REGISTRATION_FAILED for $tag"
      conda deactivate
    else
      echo "rdsm exists, skip registration"
    fi
    echo "===== DONE $tag $(date '+%F %T') ====="
  done
done
echo "===== M2B SATNERF CKPT ALL DONE $(date '+%F %T') ====="
