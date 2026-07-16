import sys
import nibabel as nib
import numpy as np


gm_file=sys.argv[1]
wm_file=sys.argv[2]
csf_file=sys.argv[3]
ref_file=sys.argv[4]
out_file=sys.argv[5]


gm=nib.load(gm_file).get_fdata()
wm=nib.load(wm_file).get_fdata()
csf=nib.load(csf_file).get_fdata()


prob=np.stack(
    [csf,gm,wm],
    axis=0
)


seg=np.argmax(prob,axis=0)+1


ref=nib.load(ref_file)


out=nib.Nifti1Image(
    seg.astype("uint8"),
    ref.affine,
    ref.header
)


nib.save(out,out_file)

print("Saved:",out_file)
