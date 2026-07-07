1. Read raw using data_read_code and convert to nii.gz. or use read_raw_roll2npy.
2. Go to data prep; Resample to 2x2x5 mm3 and save to resample.
3. Read nii.gz from resample and perform roll and allign the axis and shapes.
4. Save to .npy.
5. Go to Local folder; Make same size of x,y,z axis and in xyz as axis in each one for NiftyMIC.