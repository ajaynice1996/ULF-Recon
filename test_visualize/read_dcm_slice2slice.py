import os
import pydicom


def print_mri_summary(ds, fname):
    """
    Print important MRI acquisition parameters in a clean format.
    """
    print("\n--- MRI SUMMARY ---")
    print(f"File: {fname}")

    keys = [
        "PatientName",
        "PatientID",
        "StudyDate",
        "Modality",
        "Manufacturer",
        "ManufacturerModelName",
        "MagneticFieldStrength",
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "Rows",
        "Columns",
        "PixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "RepetitionTime",
        "EchoTime",
        "FlipAngle",
        "ImageOrientationPatient",
        "ImagePositionPatient",
    ]

    for k in keys:
        if hasattr(ds, k):
            print(f"{k:30}: {getattr(ds, k)}")


def read_dicom_folder(folder_path, print_full_tags=True):
    """
    Read all DICOM files in a folder and display metadata.
    """

    dcm_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ])

    if not dcm_files:
        print("No .dcm files found in folder.")
        return

    print(f"\nFound {len(dcm_files)} DICOM files in:\n{folder_path}\n")

    for fname in dcm_files:
        fpath = os.path.join(folder_path, fname)

        try:
            ds = pydicom.dcmread(fpath)

            print("\n" + "=" * 100)
            print(f"FILE: {fname}")
            print("=" * 100)

            # FULL DICOM TAGS
            if print_full_tags:
                print("\n--- FULL DICOM METADATA ---")
                print(ds)

            # CLEAN SUMMARY
            print_mri_summary(ds, fname)

        except Exception as e:
            print(f"\nError reading {fname}: {e}")


# ============================
# USAGE
# ============================

if __name__ == "__main__":

    # import pydicom

    # ds = pydicom.dcmread("DataSRR/BT/BT_0001/BT_KKI_0001_3T_S1/std_20260218_154201621/MRe.1.3.46.670589.11.45002.5.20.1.1.10980.2026021815170916108.dcm")

    # print("PixelData:", hasattr(ds, "PixelData"))
    # print("NumberOfFrames:", getattr(ds, "NumberOfFrames", None))

    folder = "DataSRR/BT/BT_0001/BT_KKI_0001_3T_S1/std_20260218_154201621"   # <-- CHANGE THIS
    read_dicom_folder(folder, print_full_tags=True)