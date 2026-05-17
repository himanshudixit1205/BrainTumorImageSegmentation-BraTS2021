import os
import random
import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tqdm import tqdm

from utils.utils import (
    focal_tversky_loss,
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

# ---------------- PATHS ----------------

#  ---------------- BEST V3 MODEL  ----------------
MODEL_PATH = "../models/focal_tversky_brats/best_focal_brats.keras"

# ---------------- FULL TCGA-LGG DATASET ----------------
TCGA_ROOTS = [
    r"datasets/kaggle_3m/train",
    r"datasets/kaggle_3m/validation",
    r"datasets/kaggle_3m/test"
]

OUTPUT_CSV = "../results/metrics/tcga_lgg_external_validation.csv"

# ---------------- LOAD MODEL
print("\nLoading model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "focal_tversky_loss": focal_tversky_loss,
        "dice_coefficients": dice_coefficients,
        "iou": iou,
        "precision_m": precision_m,
        "recall_m": recall_m
    }
)

print(" Model loaded successfully")

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

# ---------------- MASK LOADING ----------------
def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(f" Could not load mask: {path}")

    mask = cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # ---------------- Convert to binary mask ----------------
    mask = (mask > 0).astype(np.uint8)

    return mask

# ---------------- BUILD DATAFRAME ----------------
print("\nBuilding TCGA-LGG dataframe...")

rows = []

for tcga_root in TCGA_ROOTS:

    print(f"\nScanning folder: {tcga_root}")

    patient_dirs = sorted([
        d for d in os.listdir(tcga_root)
        if os.path.isdir(os.path.join(tcga_root, d))
    ])

    print(f"Patients found: {len(patient_dirs)}")

    for patient_id in patient_dirs:

        patient_path = os.path.join(
            tcga_root,
            patient_id
        )

        files = os.listdir(patient_path)

        image_files = sorted([
            f for f in files
            if (
                f.endswith(".tif") and
                "_mask" not in f
            )
        ])

        for image_file in image_files:

            mask_file = image_file.replace(
                ".tif",
                "_mask.tif"
            )

            image_path = os.path.join(
                patient_path,
                image_file
            )

            mask_path = os.path.join(
                patient_path,
                mask_file
            )

            if not os.path.exists(mask_path):
                continue

            rows.append({

                "patient_id": patient_id,
                "image": image_path,
                "mask": mask_path
            })

# ---------------- CREATE DATAFRAME ----------------
tcga_df = pd.DataFrame(rows)

print("\n" + "=" * 60)
print("TCGA-LGG DATASET SUMMARY")
print("=" * 60)

print(f"Total slices   : {len(tcga_df)}")
print(f"Total patients : {tcga_df['patient_id'].nunique()}")

print("=" * 60)

# ---------------- RESULTS STORAGE ----------------
results = []

# ---------------- PATIENT LOOP ----------------
print("\nRunning external validation...")

unique_patients = tcga_df["patient_id"].unique()

for patient_id in tqdm(unique_patients, desc="Patients"):

    patient_df = tcga_df[
        tcga_df["patient_id"] == patient_id
    ]

    gt_volume = []
    pred_volume = []

    # ---------------- SLICE LOOP ----------------
    for _, row in patient_df.iterrows():

        image_path = row["image"]
        mask_path = row["mask"]

        # ---------------- LOAD IMAGE ----------------
        img = load_image(image_path)

        img_input = np.expand_dims(
            img,
            axis=0
        )

        # ---------------- LOAD GT MASK ----------------
        gt_mask = load_mask(mask_path)

        # ---------------- Skip empty masks ----------------
        if np.sum(gt_mask) == 0:
            continue

        # ---------------- MODEL PREDICTION ----------------
        pred = model.predict(
            img_input,
            verbose=0
        )[0, ..., 0]

        pred_bin = (
            pred > 0.5
        ).astype(np.uint8)

        # ---------------- STORE ----------------
        gt_volume.append(gt_mask)

        pred_volume.append(pred_bin)

    # ---------------- SKIP EMPTY PATIENTS ----------------
    if len(gt_volume) == 0:
        continue

    # ---------------- STACK TO 3D VOLUMES ----------------
    gt_volume = np.stack(
        gt_volume,
        axis=0
    )

    pred_volume = np.stack(
        pred_volume,
        axis=0
    )

    # ---------------- COMPUTE METRICS ----------------
    dice_val, iou_val, precision_val, recall_val = (
        compute_metrics(
            gt_volume,
            pred_volume
        )
    )

    # ---------------- SAVE RESULTS ----------------
    results.append({

        "patient_id": patient_id,

        "dice": dice_val,
        "iou": iou_val,
        "precision": precision_val,
        "recall": recall_val
    })

# ---------------- SAVE CSV ----------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ---------------- FINAL RESULTS ----------------
print("\n" + "=" * 60)
print("TCGA-LGG EXTERNAL VALIDATION RESULTS")
print("=" * 60)

print(f"\nPatients evaluated: {len(results_df)}")

if len(results_df) > 0:

    print("\n---------------- MEAN METRICS ----------------")
    print(f"Mean Dice      : {results_df['dice'].mean():.4f}")
    print(f"Mean IoU       : {results_df['iou'].mean():.4f}")
    print(f"Mean Precision : {results_df['precision'].mean():.4f}")
    print(f"Mean Recall    : {results_df['recall'].mean():.4f}")

    print("\n---------------- MEDIAN METRICS ----------------")
    print(f"Median Dice      : {results_df['dice'].median():.4f}")
    print(f"Median IoU       : {results_df['iou'].median():.4f}")
    print(f"Median Precision : {results_df['precision'].median():.4f}")
    print(f"Median Recall    : {results_df['recall'].median():.4f}")

    print("\n---------------- STANDARD DEVIATION ----------------")
    print(f"Dice STD : {results_df['dice'].std():.4f}")
    print(f"IoU STD  : {results_df['iou'].std():.4f}")

print("\n Saved:", OUTPUT_CSV)
print("=" * 60)