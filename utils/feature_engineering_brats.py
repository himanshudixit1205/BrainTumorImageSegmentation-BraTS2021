import os
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
import random
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import yaml

with open("../configs/config_brats.yaml", "r") as file:
    config = yaml.safe_load(file)

# ------------------------- CONFIGURATION -------------------------
BRATS_ROOT = "../datasets/BraTS2021_Training_Data"
OUTPUT_DIR = "../new_brats_processed"
IMG_SIZE = config["training"]["image_size"]
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = config["training"]["seed"]
KEEP_EMPTY_RATIO = 0.2

CHANNEL_ORDER = ['flair', 't1ce', 't2']


# -----------------------------------------------------------------
def normalize_slice(slice_2d):
    """Z-score normalization per slice"""
    mean = np.mean(slice_2d)
    std = np.std(slice_2d)
    if std == 0:
        return np.zeros_like(slice_2d)
    return (slice_2d - mean) / std


def load_patient_niftis(patient_dir):
    """Load all modalities for a patient"""
    modalities = {}
    for mod in ['flair', 't1', 't1ce', 't2', 'seg']:
        files = glob(os.path.join(patient_dir, f"*{mod}.nii*"))
        if not files:
            raise FileNotFoundError(f"No {mod} file in {patient_dir}")
        modalities[mod] = nib.load(files[0]).get_fdata()
    return modalities


def process_patient(patient_dir, out_img_dir, out_mask_dir, patient_id, target_size):
    """Convert 3D volume into 2D slices and save"""
    
    data = load_patient_niftis(patient_dir)
    seg = data['seg']
    binary_seg = (seg > 0).astype(np.uint8)

    n_slices = seg.shape[2]
    slice_files = []

    # 🔥 reproducible RNG per patient
    rng = random.Random(RANDOM_SEED + hash(patient_id) % 10000)

    for z in range(n_slices):

        img_channels = []

        for mod in CHANNEL_ORDER:
            slice_2d = data[mod][:, :, z]
            slice_2d = normalize_slice(slice_2d)

            resized = cv2.resize(
                slice_2d,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )

            img_channels.append(resized)

        img_3ch = np.stack(img_channels, axis=-1)

        # Mask
        mask_slice = binary_seg[:, :, z]
        mask_resized = cv2.resize(
            mask_slice,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_resized = (mask_resized > 0.5).astype(np.uint8)

        # 🔥 Balance empty slices (reproducible)
        if np.sum(mask_resized) == 0 and rng.random() > KEEP_EMPTY_RATIO:
            continue

        # Normalize image to 0–255
        img_norm = img_3ch - np.min(img_3ch)
        if np.max(img_norm) != 0:
            img_norm = img_norm / np.max(img_norm)

        img_uint8 = (img_norm * 255).astype(np.uint8)

        # Save image
        img_filename = f"{patient_id}_slice{z:04d}.png"
        img_path = os.path.join(out_img_dir, img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

        # Save mask
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        mask_filename = f"{patient_id}_slice{z:04d}_mask.png"
        mask_path = os.path.join(out_mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask_uint8)

        slice_files.append((img_path, mask_path, patient_id))

    return slice_files

# ---------------- MAIN ----------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    img_dir = os.path.join(OUTPUT_DIR, "images")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # ---------------- GET PATIENT FOLDERS ----------------
    patient_folders = glob(os.path.join(BRATS_ROOT, "BraTS2021_*"))
    patient_folders = [p for p in patient_folders if os.path.isdir(p)]

    print(f"Found {len(patient_folders)} patients.")

    # ---------------- SPLIT DATA ----------------
    train_val, test = train_test_split(
        patient_folders,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED
    )

    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED
    )

    print(f"Train patients: {len(train)}")
    print(f"Val patients: {len(val)}")
    print(f"Test patients: {len(test)}")

    #  Sanity check (no overlap)
    print("\nSanity Check:")
    print("Train ∩ Val:", len(set(train) & set(val)))
    print("Train ∩ Test:", len(set(train) & set(test)))
    print("Val ∩ Test:", len(set(val) & set(test)))

    train_records, val_records, test_records = [], [], []

    # ---------------- PROCESS ----------------
    for pat_dir in tqdm(train, desc="Train patients"):
        pid = os.path.basename(pat_dir)
        train_records.extend(process_patient(pat_dir, img_dir, mask_dir, pid, TARGET_SIZE))

    for pat_dir in tqdm(val, desc="Val patients"):
        pid = os.path.basename(pat_dir)
        val_records.extend(process_patient(pat_dir, img_dir, mask_dir, pid, TARGET_SIZE))

    for pat_dir in tqdm(test, desc="Test patients"):
        pid = os.path.basename(pat_dir)
        test_records.extend(process_patient(pat_dir, img_dir, mask_dir, pid, TARGET_SIZE))

    # ---------------- SAVE CSV ----------------
    columns = ['image', 'mask', 'patient_id']

    train_df = pd.DataFrame(train_records, columns=columns)
    val_df = pd.DataFrame(val_records, columns=columns)
    test_df = pd.DataFrame(test_records, columns=columns)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("\nDone!")
    print(f"Train slices: {len(train_df)}")
    print(f"Val slices: {len(val_df)}")
    print(f"Test slices: {len(test_df)}")


if __name__ == "__main__":
    main()