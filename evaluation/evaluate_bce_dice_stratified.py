import os
import math
import random
import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tqdm import tqdm

from utils.utils import (
    bce_dice_loss,
    dice_coefficients,
    iou,
    precision_m,
    recall_m,
    train_generator_shuffle,
    volume_dice, 
    volume_iou, 
    volume_precision, 
    volume_recall, 
    volume_specificity
)

# ---------------- GPU SETTINGS ----------------
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ---------------- LOAD CONFIG ----------------
with open("../configs/config_brats.yaml", "r") as file:
    config = yaml.safe_load(file)

SEED = config["training"]["seed"]
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE = config["training"]["batch_size"]
LR = config["training"]["learning_rate"]
EPOCHS = config["training"]["epochs"]
IMG_SIZE = config["training"]["image_size"]

MODEL_PATH = "../models/bce_dice_brats/best_bce_dice_brats.keras"
TEST_CSV = "../new_brats_processed/test.csv"

SAVE_PREDICTIONS = False
OUTPUT_PREDS_DIR = "test_predictions"

# ---------------- LOAD MODEL ----------------
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficients": dice_coefficients,
        "iou": iou,
        "precision_m": precision_m,
        "recall_m": recall_m
    }
)
print(" Model loaded successfully")

# ---------------- LOAD TEST CSV ----------------
test_df = pd.read_csv(TEST_CSV)
print("\nTest columns:", test_df.columns)
print(f"\nTotal test slices: {len(test_df)}")

if "patient_id" not in test_df.columns:
    raise ValueError(" test.csv must contain 'patient_id' column")

slice_patient_ids = test_df["patient_id"].values
unique_patients = test_df["patient_id"].nunique()
print(f"Total test patients: {unique_patients}")

if SAVE_PREDICTIONS:
    os.makedirs(OUTPUT_PREDS_DIR, exist_ok=True)

# ---------------- CREATE TEST GENERATOR ----------------
test_gen = train_generator_shuffle(
    test_df,
    batch_size=BATCH_SIZE,
    augmentation_dict={},
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    seed=SEED
)

steps = math.ceil(len(test_df) / BATCH_SIZE)

# ---------------- STORE PATIENT DATA ----------------
patient_data = {}
slice_idx = 0

print("\nRunning inference on test set...")
for batch_idx in tqdm(range(steps), desc="Batches"):
    try:
        imgs, masks = next(test_gen)
    except StopIteration:
        break

    preds = model.predict_on_batch(imgs)
    preds_bin = (preds > 0.5).astype(np.float32)

    for i in range(imgs.shape[0]):
        if slice_idx >= len(slice_patient_ids):
            break
        patient_id = slice_patient_ids[slice_idx]
        mask_slice = masks[i, ..., 0]
        pred_slice = preds_bin[i, ..., 0]

        if patient_id not in patient_data:
            patient_data[patient_id] = {"masks": [], "preds": []}
        patient_data[patient_id]["masks"].append(mask_slice)
        patient_data[patient_id]["preds"].append(pred_slice)

        if SAVE_PREDICTIONS:
            pred_vis = (pred_slice * 255).astype(np.uint8)
            save_path = os.path.join(OUTPUT_PREDS_DIR, f"{patient_id}_slice{slice_idx:04d}.png")
            cv2.imwrite(save_path, pred_vis)

        slice_idx += 1

print(f"\n Processed {slice_idx} slices")
print(f" Found {len(patient_data)} patients")

# ========== SIZE‑STRATIFIED METRICS =============

# ---- 1. Collect all slices from all patients ----
all_masks_slices = []
all_preds_slices = []
for pid, data in patient_data.items():
    all_masks_slices.extend(data["masks"])
    all_preds_slices.extend(data["preds"])

all_masks_slices = np.stack(all_masks_slices, axis=0)   # (num_slices, H, W)
all_preds_slices = np.stack(all_preds_slices, axis=0)

print(f"\nTotal slices for stratified analysis: {all_masks_slices.shape[0]}")

# ---- 2. Compute tumor size per slice (pixel count) ----
tumor_sizes_per_slice = np.sum(all_masks_slices, axis=(1,2))

# Use percentiles for thresholds (only non‑zero tumors to avoid zeros)
non_zero_sizes = tumor_sizes_per_slice[tumor_sizes_per_slice > 0]
if len(non_zero_sizes) > 0:
    p33 = np.percentile(non_zero_sizes, 33)
    p66 = np.percentile(non_zero_sizes, 66)
else:
    p33, p66 = 500, 2000   # fallback

print(f"Slice thresholds: small ≤ {int(p33)} px, medium {int(p33)}–{int(p66)} px, large > {int(p66)} px")

# ---- 3. Vectorised stratified metrics function ----
def stratified_metrics_slice(y_true, y_pred, thresholds):
    # y_true, y_pred: (N, H, W)
    sizes = np.sum(y_true, axis=(1,2))
    small_mask = (sizes <= thresholds[0]) & (sizes > 0)
    med_mask   = (sizes > thresholds[0]) & (sizes <= thresholds[1])
    large_mask = sizes > thresholds[1]

    results = {}
    for name, mask in zip(['small', 'medium', 'large'], [small_mask, med_mask, large_mask]):
        if np.sum(mask) == 0:
            results[name] = {'dice': np.nan, 'iou': np.nan, 'recall': np.nan, 'count': 0}
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        intersection = np.sum(y_t * y_p, axis=(1,2))
        sum_t = np.sum(y_t, axis=(1,2))
        sum_p = np.sum(y_p, axis=(1,2))
        dice = (2 * intersection + 1e-6) / (sum_t + sum_p + 1e-6)
        iou  = (intersection + 1e-6) / (sum_t + sum_p - intersection + 1e-6)
        recall = (intersection + 1e-6) / (sum_t + 1e-6)
        results[name] = {
            'dice': np.mean(dice),
            'iou': np.mean(iou),
            'recall': np.mean(recall),
            'count': int(np.sum(mask))
        }
    return results

strat_slice = stratified_metrics_slice(all_masks_slices, all_preds_slices, thresholds=(p33, p66))

print("\n" + "="*60)
print("SIZE‑STRATIFIED METRICS (Slice‑level)")
print("="*60)
for size in ['small', 'medium', 'large']:
    s = strat_slice[size]
    print(f"{size.upper()}: Dice={s['dice']:.4f}, IoU={s['iou']:.4f}, Recall={s['recall']:.4f} (n={s['count']} slices)")

# ---- 4. Patient‑level stratified metrics ----
patient_volumes = []
patient_volume_metrics = []   # store (volume_mask, volume_pred, total_tumor_voxels)

for pid, data in patient_data.items():
    vol_mask = np.stack(data["masks"], axis=0)
    vol_pred = np.stack(data["preds"], axis=0)
    total_tumor = np.sum(vol_mask)
    if total_tumor == 0:
        continue   # skip empty patients for stratified analysis
    patient_volumes.append(total_tumor)
    patient_volume_metrics.append((vol_mask, vol_pred, total_tumor))

if len(patient_volumes) > 0:
    p33_vol = np.percentile(patient_volumes, 33)
    p66_vol = np.percentile(patient_volumes, 66)
    print(f"\nPatient volume thresholds: small ≤ {int(p33_vol)} px³, medium {int(p33_vol)}–{int(p66_vol)} px³, large > {int(p66_vol)} px³")

    vol_results = {'small': {'dice': [], 'iou': [], 'recall': []},
                   'medium': {'dice': [], 'iou': [], 'recall': []},
                   'large': {'dice': [], 'iou': [], 'recall': []}}

    for vol_mask, vol_pred, sz in patient_volume_metrics:
        # 3D volume metrics
        intersection = np.sum(vol_mask * vol_pred)
        sum_t = np.sum(vol_mask)
        sum_p = np.sum(vol_pred)
        dice = (2 * intersection + 1e-6) / (sum_t + sum_p + 1e-6)
        iou  = (intersection + 1e-6) / (sum_t + sum_p - intersection + 1e-6)
        recall = (intersection + 1e-6) / (sum_t + 1e-6)

        if sz <= p33_vol:
            vol_results['small']['dice'].append(dice)
            vol_results['small']['iou'].append(iou)
            vol_results['small']['recall'].append(recall)
        elif sz <= p66_vol:
            vol_results['medium']['dice'].append(dice)
            vol_results['medium']['iou'].append(iou)
            vol_results['medium']['recall'].append(recall)
        else:
            vol_results['large']['dice'].append(dice)
            vol_results['large']['iou'].append(iou)
            vol_results['large']['recall'].append(recall)

    print("\n" + "="*60)
    print("SIZE‑STRATIFIED METRICS (Patient‑level volume)")
    print("="*60)
    for size in ['small', 'medium', 'large']:
        d = vol_results[size]
        if len(d['dice']) == 0:
            print(f"{size.upper()}: No patients")
        else:
            print(f"{size.upper()}: Dice={np.mean(d['dice']):.4f}, IoU={np.mean(d['iou']):.4f}, Recall={np.mean(d['recall']):.4f} (n={len(d['dice'])})")
else:
    print("\nNo non‑empty patients for volume stratification")

# ---------------- EVALUATE PATIENTS ----------------
results = []
skipped_patients = 0

print("\nComputing patient-level metrics...")
for patient_id, data in tqdm(patient_data.items(), desc="Patients"):
    vol_masks = np.stack(data["masks"], axis=0)
    vol_preds = np.stack(data["preds"], axis=0)

    if np.sum(vol_masks) == 0 and np.sum(vol_preds) == 0:
        skipped_patients += 1
        continue

    dice_val = volume_dice(vol_masks, vol_preds)
    iou_val = volume_iou(vol_masks, vol_preds)
    precision_val = volume_precision(vol_masks, vol_preds)
    recall_val = volume_recall(vol_masks, vol_preds)
    specificity_val = volume_specificity(vol_masks, vol_preds)

    results.append({
        "patient_id": patient_id,
        "dice": dice_val,
        "iou": iou_val,
        "precision": precision_val,
        "recall": recall_val,
        "specificity": specificity_val
    })

results_df = pd.DataFrame(results)

# ---------------- FINAL RESULTS ----------------
print("\n" + "=" * 60)
print("FINAL PATIENT-LEVEL TEST RESULTS")
print("=" * 60)
print(f"Evaluated Patients : {len(results_df)}")
print(f"Skipped Patients   : {skipped_patients}")

print("\n---------------- MEAN METRICS ----------------")
print(f"Mean Dice Score : {results_df['dice'].mean():.4f}")
print(f"Mean IoU        : {results_df['iou'].mean():.4f}")
print(f"Mean Precision  : {results_df['precision'].mean():.4f}")
print(f"Mean Recall     : {results_df['recall'].mean():.4f}")
print(f"Mean Specificity: {results_df['specificity'].mean():.4f}")

print("\n---------------- STANDARD DEVIATION ----------------")
print(f"Dice STD : {results_df['dice'].std():.4f}")
print(f"IoU STD  : {results_df['iou'].std():.4f}")

print("\n---------------- MEDIAN METRICS ----------------")
print(f"Median Dice Score : {results_df['dice'].median():.4f}")
print(f"Median IoU        : {results_df['iou'].median():.4f}")
print(f"Median Precision  : {results_df['precision'].median():.4f}")
print(f"Median Recall     : {results_df['recall'].median():.4f}")
print(f"Median Specificity: {results_df['specificity'].median():.4f}")
print("=" * 60)

results_df.to_csv("../results/metrics/bce_dice_stratified_metrics.csv", index=False)
print("\n Saved: bce_dice_stratified_metrics.csv")
