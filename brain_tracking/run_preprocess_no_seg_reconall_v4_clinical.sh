#!/usr/bin/env bash
set -euo pipefail

# Clinical-style variant with cross-stack harmonization step.

PYTHON_BIN="/Users/ashar126/Documents/Development/Work/ULF-Recon/.venv-1/bin/python"
SCRIPT_PATH="brain_tracking/preprocess_stacks_for_niftymic_no_segmentation_reconall_v4_clinical.py"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --input-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40" \
  --output-dir "DataSRR/volunteer_xxx/10mm/npy/native_niftymic_inputs_thr40/no_seg_preproc_reconall_v4_clinical" \
  --fixed-fig-dir "DataSRR/volunteer_xxx/figures/no_seg_preproc_reconall_v4_clinical"
