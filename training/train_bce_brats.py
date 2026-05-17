import os
import math
import random
import numpy as np
import tensorflow as tf
import keras
import yaml
import pandas as pd
from utils.utils import *
from utils.unet import unet

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

# ---------------- LOAD CSV ----------------
train_df = pd.read_csv("../new_brats_processed/train.csv")
val_df   = pd.read_csv("../new_brats_processed/val.csv")

print("Train columns:", train_df.columns)
print("Val columns:", val_df.columns)

# ---------------- PATIENT SAFETY CHECK ----------------
train_ids = set(train_df["patient_id"])
val_ids   = set(val_df["patient_id"])

overlap = train_ids & val_ids

print("Train patients:", len(train_ids))
print("Val patients:", len(val_ids))

if overlap:
    print(f" ERROR: {len(overlap)} overlapping patients!")
    exit(1)

print(" No leakage detected")

# ---------------- PATIENT-AWARE SUBSAMPLING ----------------
sub_frac = config["training"].get("subsample_fraction", 1.0)

if sub_frac < 1.0:
    unique_patients = train_df["patient_id"].unique()

    sampled_patients = np.random.choice(
        unique_patients,
        int(len(unique_patients) * sub_frac),
        replace=False
    )

    train_df = train_df[train_df["patient_id"].isin(sampled_patients)].reset_index(drop=True)

# ---------------- LIMIT VALIDATION SIZE ----------------
MAX_VAL_SAMPLES = config['training']['MAX_VAL_SAMPLES']
if len(val_df) > MAX_VAL_SAMPLES:
    val_df = val_df.sample(n=MAX_VAL_SAMPLES, random_state=SEED).reset_index(drop=True)

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")

# ---------------- GENERATORS ----------------
train_gen = train_generator_shuffle(
    train_df, 
    batch_size=BATCH_SIZE,
    augmentation_dict=config["augmentation"],
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=SEED
)

val_gen = train_generator_shuffle(
    val_df,
    batch_size=BATCH_SIZE,
    augmentation_dict={},
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    seed=SEED
)

# ---------------- DEBUG ----------------
x_batch, y_batch = next(train_gen)
print("Image range:", np.min(x_batch), np.max(x_batch))
print("Mask range:", np.min(y_batch), np.max(y_batch))
print("Mask unique:", np.unique(y_batch))
if np.max(y_batch) == 0:
    print(" WARNING: First batch contains all empty masks! Shuffling may not work correctly.")
print("Image shape:", x_batch.shape)
print("Mask shape:", y_batch.shape)

# ---------------- MODEL ----------------
model = unet((IMG_SIZE, IMG_SIZE, 3))

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss=bce_loss,
    metrics=[
        dice_coefficients,
        iou,
        precision_m,
        recall_m
    ]
)

# ---------------- CALLBACKS ----------------
os.makedirs("../models/bce_brats", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "../models/bce_brats/best_bce_brats.keras",
        monitor="val_dice_coefficients",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    SaveBestEpoch(),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coefficients',
        factor=config["callbacks"]["reduce_lr_factor"],
        patience=config["callbacks"]["reduce_lr_patience"],
        min_lr=config["callbacks"]["min_lr"],
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficients',
        mode='max',
        patience=config["callbacks"]["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1
    )
]

# ---------------- TRAIN ----------------
train_steps = min(config['training']['train_steps'], math.ceil(len(train_df) / BATCH_SIZE))
val_steps   = min(config['training']['val_steps'], math.ceil(len(val_df) / BATCH_SIZE))

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks
)

# ---------------- SAVE ----------------
model.save("../models/bce_brats/final_bce_brats.keras.keras")