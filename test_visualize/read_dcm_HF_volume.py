import os
import numpy as np
import pydicom
from nibabel.viewers import OrthoSlicer3D


def print_mri_summary(ds, fname):
    print("\n--- MRI SUMMARY ---")
    print(f"File: {fname}")

    keys = [
        "PatientName", "PatientID", "StudyDate", "Modality",
        "Manufacturer", "ManufacturerModelName",
        "MagneticFieldStrength",
        "SeriesDescription", "ProtocolName", "SequenceName",
        "Rows", "Columns",
        "PixelSpacing", "SliceThickness",
        "SpacingBetweenSlices",
        "RepetitionTime", "EchoTime", "FlipAngle",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "NumberOfFrames"
    ]

    for k in keys:
        if hasattr(ds, k):
            print(f"{k:30}: {getattr(ds, k)}")


import numpy as np
import pydicom


def load_multiframe_dicom(dcm_path):
    """
    Load a single multi-frame DICOM file as a 3D volume.
    """

    ds = pydicom.dcmread(dcm_path)

    print("\n===== DICOM INFO =====")
    print("File:", dcm_path)
    print("Rows:", getattr(ds, "Rows", None))
    print("Columns:", getattr(ds, "Columns", None))
    print("NumberOfFrames:", getattr(ds, "NumberOfFrames", None))

    if not hasattr(ds, "NumberOfFrames") or ds.NumberOfFrames <= 1:
        raise ValueError("This is NOT a multi-frame DICOM (not a 3D volume file).")

    # (frames, H, W)
    vol = ds.pixel_array.astype(np.float32)

    print("Raw shape (frames, H, W):", vol.shape)

    # convert → (H, W, frames)
    vol = np.transpose(vol, (1, 2, 0))

    print("Final volume shape (H, W, D):", vol.shape)

    return vol, ds


def read_dicom_folder(folder_path):
    """
    Handles:
    1. Multi-frame DICOM (single file volume)
    2. Classic DICOM series (multiple slices)
    """

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ])

    if not files:
        print("No DICOM files found.")
        return

    print(f"\nFound {len(files)} DICOM files\n")

    volumes = []

    for fname in files:
        fpath = os.path.join(folder_path, fname)

        try:
            ds = pydicom.dcmread(fpath)

            print("\n" + "=" * 80)
            print(f"FILE: {fname}")
            print("=" * 80)

            print_mri_summary(ds, fname)

            # =========================
            # CASE 1: MULTI-FRAME DICOM
            # =========================
            if hasattr(ds, "NumberOfFrames") and ds.NumberOfFrames > 1:
                vol = load_multiframe(ds)
                volumes.append(vol)

            # =========================
            # CASE 2: SINGLE SLICE
            # =========================
            elif hasattr(ds, "PixelData"):
                vol = ds.pixel_array
                if vol.ndim == 2:
                    vol = vol[:, :, None]  # make it 3D
                volumes.append(vol)

        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # =========================
    # FINAL MERGE (if needed)
    # =========================
    if len(volumes) == 1:
        final_vol = volumes[0]
    else:
        final_vol = np.concatenate(volumes, axis=2)

    print("\n==============================")
    print("FINAL VOLUME SHAPE:", final_vol.shape)
    print("==============================")

    return final_vol


# ============================
# USAGE
# ============================

if __name__ == "__main__":

    # folder = "DataSRR/BT/BT_0001/BT_KKI_0001_3T_S1/std_20260218_154201621"

    dcm_file = "DataSRR/BT/BT_0001/BT_KKI_0001_3T_S1/std_20260218_154201621/MRe.1.3.46.670589.11.45002.5.20.1.1.10980.2026021815170916108.dcm"
    volume, ds = load_multiframe_dicom(dcm_file)

    # display the volume shape
    print("Volume shape:", volume.shape)

    # DISPLAY VOLUME USING ORTHOSLICER3D

    s = OrthoSlicer3D(volume)
    s.show()