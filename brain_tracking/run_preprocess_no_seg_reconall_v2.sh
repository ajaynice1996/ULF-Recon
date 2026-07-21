#!/usr/bin/env bash
set -euo pipefail

# Standalone runner for reconall-style no-seg preprocessing script.
# This does not touch previous scripts or their output folders.

PYTHON_BIN="/Users/ashar126/Documents/Development/Work/ULF-Recon/.venv-1/bin/python"
SCRIPT_PATH="brain_tracking/preprocess_stacks_for_niftymic_no_segmentation_reconall_v2.py"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --input-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40" \
  --output-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/no_seg_preproc_reconall_v2" \
  --fixed-fig-dir "DataSRR/volunteer_xxx/figures/no_seg_preproc_reconall_v2"
