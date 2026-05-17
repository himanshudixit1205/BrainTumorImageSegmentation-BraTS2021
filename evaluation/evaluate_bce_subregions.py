import os
import re
import random
import yaml
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tqdm import tqdm

from utils import (
    bce_loss,   # CHANGE FOR v2/v3
    dice_coefficients,
    iou,
    precision_m,
    recall_m
)

# ---------------- GPU SETTINGS ----------------
gpu_devices = tf.config.experimental.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ---------------- LOAD CONFIG ----------------
with open("../configs/config_brats.yaml", "r") as file:
    config = yaml.safe_load(file)

SEED = config["training"]["seed"]
IMG_SIZE = config["training"]["image_size"]

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

MODEL_PATH = "../models/bce_brats/best_bce_brats.keras"
TEST_CSV = "../new_brats_processed/test.csv"


ORIGINAL_BRATS_DIR = r"datasets/BraTS2021_Training_Data"

OUTPUT_CSV = "../results/metrics/bce_subregion_metrics.csv.csv"

# ---------------- LOAD MODEL ----------------
print("\nLoading model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_loss": bce_loss,
        "dice_coefficients": dice_coefficients,
        "iou": iou,
        "precision_m": precision_m,
        "recall_m": recall_m
    }
)

print(" Model loaded successfully")

# ---------------- LOAD TEST CSV ----------------
test_df = pd.read_csv(TEST_CSV)

print(f"\nTotal test slices: {len(test_df)}")

if "patient_id" not in test_df.columns:
    raise ValueError(" patient_id column missing in test.csv")

# ---------------- IMAGE LOADING ----------------
def load_image(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f" Could not load image: {path}")

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32) / 255.0

    return img

# ---------------- LOAD NIFTI SEGMENTATION ----------------
def load_segmentation_volume(patient_id):

    seg_path = os.path.join(
        ORIGINAL_BRATS_DIR,
        patient_id,
        f"{patient_id}_seg.nii.gz"
    )

    if not os.path.exists(seg_path):

        raise FileNotFoundError(
            f" Segmentation not found:\n{seg_path}"
        )

    seg_volume = nib.load(seg_path).get_fdata()

    return seg_volume

# ---------------- EXTRACT SLICE NUMBER ----------------
def extract_slice_index(filename):

    match = re.search(r"slice(\d+)", filename)

    if match is None:
        raise ValueError(
            f" Could not parse slice index from:\n{filename}"
        )

    return int(match.group(1))

# ---------------- VERIFY LABELS ----------------
print("\nChecking original segmentation labels...")

sample_patient = test_df.iloc[0]["patient_id"]

sample_seg = load_segmentation_volume(sample_patient)

print("Unique labels found:")
print(np.unique(sample_seg))

if not any(v in np.unique(sample_seg) for v in [2,4]):

    print("\n Invalid multiclass segmentation.")
    exit()

print("\n Original multiclass masks verified.")

# ---------------- RESULTS STORAGE ----------------
results = []

unique_patients = test_df["patient_id"].unique()

print(f"\nTotal patients: {len(unique_patients)}")

# ---------------- MAIN LOOP  ----------------
print("\nRunning sub-region evaluation...")

for patient_id in tqdm(unique_patients, desc="Patients"):

    patient_df = test_df[
        test_df["patient_id"] == patient_id
    ]

    # ---------------- LOAD FULL SEGMENTATION VOLUME ONCE  ----------------
    seg_volume = load_segmentation_volume(
        patient_id
    )

    et_gt_volume = []
    tc_gt_volume = []

    pred_volume = []

    # ---------------- SLICE LOOP  ----------------
    for _, row in patient_df.iterrows():

        image_path = row["image"]

        mask_filename = os.path.basename(
            row["mask"]
        )

        # ---------------- GET SLICE INDEX  ----------------
        slice_idx = extract_slice_index(
            mask_filename
        )

        # ---------------- LOAD IMAGE  ----------------
        img = load_image(image_path)

        img_input = np.expand_dims(
            img,
            axis=0
        )

        # ---------------- MODEL PREDICTION ----------------
        pred = model.predict(
            img_input,
            verbose=0
        )[0, ..., 0]

        pred_bin = (
            pred > 0.5
        ).astype(np.uint8)

        # ---------------- EXTRACT ORIGINAL GT SLICE ----------------
        gt_slice = seg_volume[:, :, slice_idx]

        gt_slice = cv2.resize(
            gt_slice,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        gt_slice = gt_slice.astype(np.uint8)

        # ---------------- CREATE SUBREGIONS ----------------

        # Enhancing Tumour
        gt_enhancing = (
            gt_slice == 4
        ).astype(np.uint8)

        # Tumour Core
        gt_core = (
            (gt_slice == 1) |
            (gt_slice == 4)
        ).astype(np.uint8)

        # ---------------- SKIP EMTPY SLICES ----------------
        if (
            np.sum(gt_enhancing) == 0 and
            np.sum(gt_core) == 0
        ):
            continue

        # ---------------- STORE ----------------
        et_gt_volume.append(
            gt_enhancing
        )

        tc_gt_volume.append(
            gt_core
        )

        pred_volume.append(
            pred_bin
        )

    # ---------------- SKIP EMPTY PATIENTS ----------------
    if len(pred_volume) == 0:
        continue

    # ---------------- STACK VOLUMES ----------------
    et_gt_volume = np.stack(
        et_gt_volume,
        axis=0
    )

    tc_gt_volume = np.stack(
        tc_gt_volume,
        axis=0
    )

    pred_volume = np.stack(
        pred_volume,
        axis=0
    )

    # ---------------- ET METRICS ----------------
    et_dice, et_iou, et_precision, et_recall = (
        compute_metrics(
            et_gt_volume,
            pred_volume
        )
    )

    # ---------------- TC METRICS ----------------
    tc_dice, tc_iou, tc_precision, tc_recall = (
        compute_metrics(
            tc_gt_volume,
            pred_volume
        )
    )

    # ---------------- SAVE RESULTS ----------------
    results.append({

        "patient_id": patient_id,

        "et_dice": et_dice,
        "et_iou": et_iou,
        "et_precision": et_precision,
        "et_recall": et_recall,

        "tc_dice": tc_dice,
        "tc_iou": tc_iou,
        "tc_precision": tc_precision,
        "tc_recall": tc_recall
    })

# ---------------- SAVE CSV ----------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ---------------- FINAL RESULTS ----------------
print("\n" + "=" * 60)
print("SUB-REGION METRICS RESULTS")
print("=" * 60)

print(f"\nPatients evaluated: {len(results_df)}")

if len(results_df) > 0:

    print("\n------------- ENHANCING TUMOUR (ET) -------------")
    print(f"Mean ET Dice      : {results_df['et_dice'].mean():.4f}")
    print(f"Mean ET IoU       : {results_df['et_iou'].mean():.4f}")
    print(f"Mean ET Precision : {results_df['et_precision'].mean():.4f}")
    print(f"Mean ET Recall    : {results_df['et_recall'].mean():.4f}")

    print("\n------------- TUMOUR CORE (TC) ------------------")
    print(f"Mean TC Dice      : {results_df['tc_dice'].mean():.4f}")
    print(f"Mean TC IoU       : {results_df['tc_iou'].mean():.4f}")
    print(f"Mean TC Precision : {results_df['tc_precision'].mean():.4f}")
    print(f"Mean TC Recall    : {results_df['tc_recall'].mean():.4f}")

print("\n Saved:", OUTPUT_CSV)
print("=" * 60)