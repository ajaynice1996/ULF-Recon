cd /Users/ashar126/Documents/Development/Work/ULF-Recon

source .venv-uslr/bin/activate

export PYTHONPATH=/Users/ashar126/Documents/Development/Work/ULF-Recon/USLR
export BIDS_DIR=/Users/ashar126/Documents/Development/Work/ULF-Recon/psilo-data/subject-001/

export NEURITE_BACKEND=tensorflow

cd USLR

bash scripts/preprocess.sh

python scripts/linear_registration.py

bash scripts/run_nonlinear_registration.sh




cd /Users/ashar126/Documents/Development/Work/ULF-Recon

# 1) Activate USLR env
source .venv-uslr/bin/activate

# 2) Build a minimal BIDS rawdata layout for your 3 longitudinal scans
mkdir -p psilo-data/rawdata/sub-001/ses-01/anat
mkdir -p psilo-data/rawdata/sub-001/ses-02/anat
mkdir -p psilo-data/rawdata/sub-001/ses-03/anat

cp psilo-data/subject-001/image57_MPRAGE_1.0iso_SAG_Do_MPRs_20251114143054_2_Eq_1_defaced_v1.nii.gz \
   psilo-data/rawdata/sub-001/ses-01/anat/sub-001_ses-01_run-01_T1w.nii.gz

cp psilo-data/subject-001/image59_MPRAGE_1.0iso_SAG_Do_MPRs_20251217154705_2_defaced_v3.nii.gz \
   psilo-data/rawdata/sub-001/ses-02/anat/sub-001_ses-02_run-01_T1w.nii.gz

cp psilo-data/subject-001/image7_MPRAGE_1.0iso_SAG_Do_MPRs_20251217154705_2_defaced_v2.nii.gz \
   psilo-data/rawdata/sub-001/ses-03/anat/sub-001_ses-03_run-01_T1w.nii.gz

# 3) Export USLR variables
export PYTHONPATH=/Users/ashar126/Documents/Development/Work/ULF-Recon/USLR
export BIDS_DIR=/Users/ashar126/Documents/Development/Work/ULF-Recon/psilo-data/rawdata
export NEURITE_BACKEND=tensorflow
source /Applications/freesurfer/8.2.0/SetUpFreeSurfer.sh

# 4) Run USLR
cd USLR
python scripts/preprocess.py
python scripts/linear_registration.py
python scripts/nonlinear_registration.py



#Run sequence of commands to run USLR pipeline on a subject with 3 longitudinal scans

cd /Users/ashar126/Documents/Development/Work/ULF-Recon
source .venv-uslr/bin/activate

export PYTHONPATH=/Users/ashar126/Documents/Development/Work/ULF-Recon/USLR
export BIDS_DIR=/Users/ashar126/Documents/Development/Work/ULF-Recon/psilo-data/rawdata
export FREESURFER_HOME=/Applications/freesurfer/8.2.0
export NEURITE_BACKEND=tensorflow

cd USLR
python scripts/preprocess.py
python scripts/linear_registration.py
python scripts/nonlinear_registration.py