#!/bin/bash

set -e

############################
# USER SETTINGS
############################

SUBJECT=$1

ATLAS_DIR=$(pwd)/atlas_pipeline/atlas
INPUT_DIR=$(pwd)/atlas_pipeline/input
OUTPUT_DIR=$(pwd)/atlas_pipeline/output

DOCKER_IMAGE="antsx/ants:latest"


mkdir -p ${OUTPUT_DIR}


echo "================================"
echo "Pediatric LF MRI Atlas Pipeline"
echo "================================"


############################
# 1. Geometry validation
############################

echo "Checking input geometry..."

docker run --rm \
-v $(pwd):/data \
${DOCKER_IMAGE} \
PrintHeader /data/atlas_pipeline/input/${SUBJECT}


echo "Checking atlas geometry..."

docker run --rm \
-v $(pwd):/data \
${DOCKER_IMAGE} \
PrintHeader /data/atlas_pipeline/atlas/nihpd_asym_04.5-08.5_t2w.nii



############################
# 2. Registration
############################


echo "Running ANTs SyN registration..."


docker run --rm \
-v $(pwd):/data \
${DOCKER_IMAGE} \
antsRegistrationSyN.sh \
-d 3 \
-f /data/atlas_pipeline/input/${SUBJECT} \
-m /data/atlas_pipeline/atlas/nihpd_asym_04.5-08.5_t2w.nii \
-o /data/atlas_pipeline/output/reg_


echo "Registration completed"


############################
# 2.5 Save registered atlas image
############################

echo "Creating registered atlas T2 image for QC"


docker run --rm \
-v $(pwd):/data \
${DOCKER_IMAGE} \
antsApplyTransforms \
-d 3 \
-i /data/atlas_pipeline/atlas/nihpd_asym_04.5-08.5_t2w.nii \
-r /data/atlas_pipeline/input/${SUBJECT} \
-o /data/atlas_pipeline/output/atlas_T2_registered.nii.gz \
-n Linear \
-t /data/atlas_pipeline/output/reg_1Warp.nii.gz \
-t /data/atlas_pipeline/output/reg_0GenericAffine.mat


echo "Registered atlas saved"


############################
# 3. Transform atlas images
############################


for TYPE in gm wm csf mask

do

echo "Transforming ${TYPE}"


docker run --rm \
-v $(pwd):/data \
${DOCKER_IMAGE} \
antsApplyTransforms \
-d 3 \
-i /data/atlas_pipeline/atlas/nihpd_asym_04.5-08.5_${TYPE}.nii \
-r /data/atlas_pipeline/input/${SUBJECT} \
-o /data/atlas_pipeline/output/${TYPE}_subject.nii.gz \
-n Linear \
-t /data/atlas_pipeline/output/reg_1Warp.nii.gz \
-t /data/atlas_pipeline/output/reg_0GenericAffine.mat


done



############################
# 4. Create tissue label map
############################


echo "Creating tissue segmentation"


python3 create_segmentation.py \
${OUTPUT_DIR}/gm_subject.nii.gz \
${OUTPUT_DIR}/wm_subject.nii.gz \
${OUTPUT_DIR}/csf_subject.nii.gz \
${INPUT_DIR}/${SUBJECT} \
${OUTPUT_DIR}/tissue_segmentation.nii.gz



echo "================================"
echo "DONE"
echo "Output:"
ls ${OUTPUT_DIR}
