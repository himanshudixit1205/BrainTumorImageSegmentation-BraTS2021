# visualize_brats_predictions.py


# Each image contains:
# 1. Original MRI
# 2. Ground Truth Mask
# 3. Predicted Mask
# 4. Overlay Visualization

# Overlay colors:
# GREEN = Ground Truth
# RED   = Prediction


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

from utils import (
    focal_tversky_loss,
    dice_coefficients,
    iou,
    precision_m,
    recall_m,
    train_generator_shuffle
)

# =========================================================
# GPU SETTINGS
# =========================================================
gpu_devices = tf.config.experimental.list_physical_devices('GPU')

for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# =========================================================
# LOAD CONFIG
# =========================================================
with open("config_brats_new.yaml", "r") as file:
    config = yaml.safe_load(file)

SEED = config["training"]["seed"]

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE = config["training"]["batch_size"]
IMG_SIZE = config["training"]["image_size"]

MODEL_PATH = "new_best_models_brats/best_focal_brats.keras"
TEST_CSV = "new_brats_processed/test.csv"

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================
OUTPUT_DIR = "prediction_visualizations_v3"

BEST_DIR = os.path.join(OUTPUT_DIR, "best_cases")
WORST_DIR = os.path.join(OUTPUT_DIR, "worst_cases")
SMALL_DIR = os.path.join(OUTPUT_DIR, "small_tumor_cases")

os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(WORST_DIR, exist_ok=True)
os.makedirs(SMALL_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================
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

print("✅ Model loaded successfully")

# =========================================================
# LOAD TEST CSV
# =========================================================
test_df = pd.read_csv(TEST_CSV)

print("\nTest columns:", test_df.columns)
print(f"Total test slices: {len(test_df)}")

if "patient_id" not in test_df.columns:
    raise ValueError("❌ test.csv must contain 'patient_id' column")

# =========================================================
# GENERATOR
# =========================================================
test_gen = train_generator_shuffle(
    test_df,
    batch_size=BATCH_SIZE,
    augmentation_dict={},
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    seed=SEED
)

# =========================================================
# DICE FUNCTION
# =========================================================
def compute_dice(y_true, y_pred):

    intersection = np.sum(y_true * y_pred)

    return (
        (2.0 * intersection + 1e-6)
        /
        (np.sum(y_true) + np.sum(y_pred) + 1e-6)
    )

# =========================================================
# CREATE OVERLAY
# =========================================================
def create_overlay(image, mask, pred, alpha=0.4):

    overlay = image.copy()

    # Ground truth -> GREEN
    overlay[mask > 0.5] = [0, 255, 0]

    # Prediction -> RED
    overlay[pred > 0.5] = [255, 0, 0]

    blended = cv2.addWeighted(
        image,
        1 - alpha,
        overlay,
        alpha,
        0
    )

    return blended

# =========================================================
# STORE RESULTS
# =========================================================
results = []

# IMPORTANT FIX
steps = math.ceil(len(test_df) / BATCH_SIZE)

slice_idx = 0

print("\nRunning inference...")

for batch_idx in tqdm(range(steps), desc="Batches"):

    try:
        imgs, masks = next(test_gen)

    except StopIteration:
        break

    preds = model.predict_on_batch(imgs)

    preds_bin = (preds > 0.5).astype(np.float32)

    for i in range(imgs.shape[0]):

        if slice_idx >= len(test_df):
            break

        row = test_df.iloc[slice_idx]

        patient_id = row["patient_id"]

        image = imgs[i]
        gt_mask = masks[i, ..., 0]
        pred_mask = preds_bin[i, ..., 0]

        # Dice
        dice_score = compute_dice(gt_mask, pred_mask)

        # Tumor size
        tumor_size = np.sum(gt_mask)

        results.append({
            "dice": dice_score,
            "tumor_size": tumor_size,
            "patient_id": patient_id,
            "image": image,
            "gt_mask": gt_mask,
            "pred_mask": pred_mask
        })

        slice_idx += 1

print(f"\n✅ Processed {len(results)} slices")

# =========================================================
# SORT RESULTS
# =========================================================
results_sorted_best = sorted(
    results,
    key=lambda x: x["dice"],
    reverse=True
)

results_sorted_worst = sorted(
    results,
    key=lambda x: x["dice"]
)

# =========================================================
# SMALL TUMOR FILTER
# =========================================================
small_tumor_cases = [
    r for r in results
    if 0 < r["tumor_size"] <= 350
]

small_tumor_cases = sorted(
    small_tumor_cases,
    key=lambda x: x["dice"]
)

# =========================================================
# SAVE FUNCTION
# =========================================================
def save_case(case, out_dir, prefix, idx):

    image = case["image"]
    gt_mask = case["gt_mask"]
    pred_mask = case["pred_mask"]
    dice_score = case["dice"]
    patient_id = case["patient_id"]

    # Convert image back to uint8
    image_vis = (image * 255).astype(np.uint8)

    # Overlay
    overlay = create_overlay(
        image_vis,
        gt_mask,
        pred_mask
    )

    # Convert masks to uint8
    gt_vis = (gt_mask * 255).astype(np.uint8)
    pred_vis = (pred_mask * 255).astype(np.uint8)

    gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
    pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

    # Combine panels
    combined = np.concatenate([
        image_vis,
        gt_vis,
        pred_vis,
        overlay
    ], axis=1)

    filename = (
        f"{prefix}_{idx:02d}"
        f"_dice_{dice_score:.4f}"
        f"_{patient_id}.png"
    )

    save_path = os.path.join(out_dir, filename)

    cv2.imwrite(
        save_path,
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )

# =========================================================
# SAVE BEST CASES
# =========================================================
print("\nSaving BEST cases...")

for idx, case in enumerate(results_sorted_best[:20]):

    save_case(
        case,
        BEST_DIR,
        "best",
        idx
    )

# =========================================================
# SAVE WORST CASES
# =========================================================
print("Saving WORST cases...")

for idx, case in enumerate(results_sorted_worst[:20]):

    save_case(
        case,
        WORST_DIR,
        "worst",
        idx
    )

# =========================================================
# SAVE SMALL TUMOR FAILURES
# =========================================================
print("Saving SMALL tumor cases...")

for idx, case in enumerate(small_tumor_cases[:20]):

    save_case(
        case,
        SMALL_DIR,
        "small",
        idx
    )

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 60)

print("✅ VISUALIZATION COMPLETE")

print("=" * 60)

print(f"\nSaved directories:")

print(f"\nBEST CASES:")
print(BEST_DIR)

print(f"\nWORST CASES:")
print(WORST_DIR)

print(f"\nSMALL TUMOR FAILURES:")
print(SMALL_DIR)

print("\nEach image contains:")

print("1. Original MRI")
print("2. Ground Truth Mask")
print("3. Predicted Mask")
print("4. Overlay Visualization")

print("\nOverlay colors:")
print("GREEN = Ground Truth")
print("RED   = Prediction")

print("=" * 60)