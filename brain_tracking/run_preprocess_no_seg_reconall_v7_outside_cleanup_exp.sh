#!/usr/bin/env bash
set -euo pipefail

# v7 experiment: stronger outside speckle cleanup + cross-stack harmonization (no brain segmentation).

PYTHON_BIN="/Users/ashar126/Documents/Development/Work/ULF-Recon/.venv-1/bin/python"
SCRIPT_PATH="brain_tracking/preprocess_stacks_for_niftymic_no_segmentation_reconall_v7_outside_cleanup_exp.py"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --input-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40" \
  --output-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v7_no_seg_outside_cleanup_exp" \
  --fixed-fig-dir "DataSRR/volunteer_xxx/figures/reconall_v7_no_seg_outside_cleanup_exp"
