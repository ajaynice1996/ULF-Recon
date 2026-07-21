#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ashar126/Documents/Development/Work/ULF-Recon/.venv-1/bin/python"
SCRIPT_PATH="brain_tracking/fix_final_2mm_brain_segmentation.py"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --output-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v7_no_seg_outside_cleanup_exp" \
  --synthstrip-min-relative 0.6
