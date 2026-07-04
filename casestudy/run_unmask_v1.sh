#!/usr/bin/env bash
# Regenerate UNMASKED registered DSMs for all EOGS-v1 runs.
# The released eval_dsm.py NaNs water inside registration (pred_dsm[water_mask]=nan);
# eval_dsm_unmasked.py is identical except water_mask_path=None. Originals are kept
# as <scene>_rdsm_masked.tif; downstream tooling picks up the unmasked <scene>_rdsm.tif.
set -uo pipefail
cd /root/autodl-tmp/eogs_v1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eogsv1

for d in output/m2a_eogsv1_*; do
  [[ -d $d ]] || continue
  base=${d#output/m2a_eogsv1_}
  scene=${base%_s[0-9]*}
  dsmdir=$d/test_opNone/ours_5000/dsm
  if [[ ! -d $dsmdir ]]; then echo "SKIP no dsm: $d"; continue; fi
  dsm=$(ls "$dsmdir" | sort -V | tail -1)
  mkdir -p "$d/unmasked"
  if [[ ! -f $d/unmasked/${scene}_rdsm.tif ]]; then
    if ! python scripts/eval/eval_dsm_unmasked.py --pred-dsm-path "$dsmdir/$dsm" \
        --gt-dir "data/truth/$scene" --out-dir "$d/unmasked/" --aoi-id "$scene" >/dev/null 2>&1; then
      echo "FAIL $d"; continue
    fi
  fi
  if [[ ! -f $d/${scene}_rdsm_masked.tif ]]; then
    mv "$d/${scene}_rdsm.tif" "$d/${scene}_rdsm_masked.tif"
  fi
  cp "$d/unmasked/${scene}_rdsm.tif" "$d/${scene}_rdsm.tif"
  echo "OK $d"
done
echo "UNMASK ALL DONE"
