#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/ashar126/Documents/Development/Work/ULF-Recon/.venv-1/bin/python"
SCRIPT_PATH="brain_tracking/run_v7_no_seg_full_niftymic_pipeline.py"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --input-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40" \
  --output-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/reconall_v7_no_seg_outside_cleanup_exp" \
  --fixed-fig-dir "DataSRR/volunteer_xxx/figures/reconall_v7_no_seg_outside_cleanup_exp"
